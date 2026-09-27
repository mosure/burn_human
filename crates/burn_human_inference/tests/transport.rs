#![cfg(all(feature = "transport", not(target_arch = "wasm32")))]
use burn_human_inference::transport::{ModelSource, read_bounded};
use burn_human_motion::artifacts::{Part, PartReader, sha256};

#[test]
fn object_integrity_survives_async_digest_and_single_part_shortcut() -> anyhow::Result<()> {
    use burn_human_motion::artifacts::{Object, read_object};
    use std::{collections::VecDeque, task::Poll};

    struct RawReader(VecDeque<Vec<u8>>);
    impl PartReader for RawReader {
        async fn read_part(&mut self, _: &Part) -> anyhow::Result<Vec<u8>> {
            self.0
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("missing part"))
        }
        async fn digest(&mut self, bytes: &[u8]) -> anyhow::Result<String> {
            // Exercise a digest that really suspends, as browser crypto does.
            let mut pending = true;
            std::future::poll_fn(|cx| {
                if std::mem::take(&mut pending) {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                } else {
                    Poll::Ready(())
                }
            })
            .await;
            Ok(sha256(bytes))
        }
    }
    pollster::block_on(async {
        for chunks in [
            vec![b"abcdefgh".to_vec()],
            vec![b"abcd".to_vec(), b"efgh".to_vec()],
        ] {
            let object = Object {
                stage: "integrity".into(),
                size: 8,
                sha256: sha256(b"abcdefgh"),
                parts: chunks
                    .iter()
                    .map(|v| Part {
                        size: v.len(),
                        sha256: sha256(v),
                    })
                    .collect(),
                tensors: vec![],
            };
            let make_reader = || RawReader(chunks.clone().into());
            assert_eq!(read_object(&mut make_reader(), &object).await?, b"abcdefgh");
            let mut corrupt = make_reader();
            corrupt.0[0][0] ^= 1;
            assert!(read_object(&mut corrupt, &object).await.is_err());
            let mut truncated = make_reader();
            truncated.0[0].pop();
            assert!(read_object(&mut truncated, &object).await.is_err());
            let mut wrong_object = object.clone();
            wrong_object.sha256 = "0".repeat(64);
            assert!(
                read_object(&mut make_reader(), &wrong_object)
                    .await
                    .is_err()
            );
            if object.parts.len() > 1 {
                let mut reordered = object.clone();
                reordered.parts.reverse();
                let mut reordered_reader = make_reader();
                reordered_reader.0.make_contiguous().reverse();
                assert!(
                    read_object(&mut reordered_reader, &reordered)
                        .await
                        .is_err()
                );
            }
        }
        Ok(())
    })
}

#[test]
fn sequential_shards_reuse_one_http_connection() -> anyhow::Result<()> {
    use std::io::{BufRead, BufReader, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let url = format!("http://{}/part", listener.local_addr()?);
    let server = std::thread::spawn(move || -> anyhow::Result<()> {
        let (mut stream, _) = listener.accept()?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(3)))?;
        let mut reader = BufReader::new(stream.try_clone()?);
        for _ in 0..2 {
            loop {
                let mut line = String::new();
                anyhow::ensure!(
                    reader.read_line(&mut line)? > 0,
                    "client closed the reusable connection"
                );
                if line == "\r\n" {
                    break;
                }
            }
            stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: keep-alive\r\n\r\npart",
            )?;
            stream.flush()?;
        }
        Ok(())
    });
    pollster::block_on(async {
        for _ in 0..2 {
            assert_eq!(read_bounded(&url, 4).await?, b"part");
        }
        Ok::<_, anyhow::Error>(())
    })?;
    server.join().unwrap()?;
    Ok(())
}

#[test]
fn pinned_location_rejects_substituted_manifest_before_weights() -> anyhow::Result<()> {
    use burn_human_inference::pretrained::ArtifactLocation;
    use burn_human_motion::artifacts::{Manifest, Object, TensorSpec};
    pollster::block_on(async {
        let directory = tempfile::tempdir()?;
        let bytes = 1.0_f32.to_le_bytes();
        let mut manifest = Manifest {
            schema_version: 2,
            model: "fixture".into(),
            model_revision: "a".repeat(40),
            source_revision: "b".repeat(40),
            converter: "fixture".into(),
            license: "MIT".into(),
            config: serde_json::json!({}),
            assets: vec![],
            content_sha256: String::new(),
            objects: vec![Object {
                stage: "fixture".into(),
                sha256: sha256(&bytes),
                size: 4,
                parts: vec![Part {
                    sha256: sha256(&bytes),
                    size: 4,
                }],
                tensors: vec![TensorSpec {
                    name: "fixture".into(),
                    shape: vec![1],
                    dtype: "f32".into(),
                    sha256: sha256(&bytes),
                }],
            }],
        };
        manifest.seal()?;
        let location = ArtifactLocation {
            base: directory.path().to_string_lossy().into_owned(),
            sha256: manifest.content_sha256.clone(),
        };
        std::fs::write(
            directory.path().join("manifest.json"),
            serde_json::to_vec(&manifest)?,
        )?;
        assert_eq!(location.open().await?.0.content_sha256, location.sha256);
        manifest.config = serde_json::json!({"substituted": true});
        manifest.seal()?;
        std::fs::write(
            directory.path().join("manifest.json"),
            serde_json::to_vec(&manifest)?,
        )?;
        let error = location
            .open()
            .await
            .err()
            .expect("substituted manifest must fail");
        assert!(error.to_string().contains("identity mismatch"), "{error}");
        assert!(!directory.path().join("parts").exists());
        Ok(())
    })
}

#[test]
fn bounded_reads_and_corrupt_cache_recovery() -> anyhow::Result<()> {
    pollster::block_on(async {
        let source = tempfile::tempdir()?;
        let cache = tempfile::tempdir()?;
        let bytes = b"verified part";
        let part = Part {
            sha256: sha256(bytes),
            size: bytes.len(),
        };
        std::fs::create_dir(source.path().join("parts"))?;
        std::fs::write(source.path().join(part.path()), bytes)?;
        let mut reader = ModelSource::new(source.path().to_string_lossy().into());
        reader.cache = Some(cache.path().into());
        assert_eq!(reader.read_part(&part).await?, bytes);
        assert_eq!(reader.downloaded_bytes, bytes.len());
        assert_eq!(reader.read_part(&part).await?, bytes);
        assert_eq!(reader.cache_hits, 1);
        std::fs::write(cache.path().join(part.path()), b"corrupt part!")?;
        assert_eq!(reader.read_part(&part).await?, bytes);
        assert_eq!(reader.downloaded_bytes, bytes.len() * 2);
        let unavailable_cache = source.path().join("cache-is-a-file");
        std::fs::write(&unavailable_cache, b"not a directory")?;
        let mut without_cache = ModelSource::new(source.path().to_string_lossy().into());
        without_cache.cache = Some(unavailable_cache);
        assert_eq!(without_cache.read_part(&part).await?, bytes);
        assert_eq!(without_cache.cache_hits, 0);
        assert!(
            read_bounded(
                &source.path().join(part.path()).to_string_lossy(),
                bytes.len() - 1
            )
            .await
            .is_err()
        );
        std::fs::write(cache.path().join(part.path()), b"corrupt again")?;
        std::fs::write(source.path().join(part.path()), b"corrupt source")?;
        assert!(reader.read_part(&part).await.is_err());
        Ok(())
    })
}
