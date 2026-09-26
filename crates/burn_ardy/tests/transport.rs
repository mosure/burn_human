#![cfg(all(feature = "transport", not(target_arch = "wasm32")))]
use burn_ardy::transport::{ModelSource, read_bounded};
use burn_human_motion::artifacts::{Part, PartReader, sha256};

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
