use burn::tensor::TensorData;
use burn_human_inference::weights::read_tensors;
use burn_human_motion::artifacts::{Object, Part, PartReader, TensorSpec, sha256};
use burn_pack::Writer as BurnpackWriter;
use burn_store::bridge::from_data as pack_tensor;

struct Reader(Option<Vec<u8>>);
impl PartReader for Reader {
    async fn read_part(&mut self, _: &Part) -> anyhow::Result<Vec<u8>> {
        self.0.take().ok_or_else(|| anyhow::anyhow!("already read"))
    }
}

fn fixture(values: Vec<f32>) -> (Object, Vec<u8>) {
    let tensors = [
        ("first", TensorData::new(values, [3])),
        ("second", TensorData::new(vec![4.0f32, 5.0], [1, 2])),
    ];
    let specs = tensors
        .iter()
        .map(|(name, data)| TensorSpec {
            name: (*name).into(),
            shape: data.shape().to_vec(),
            dtype: "f32".into(),
            sha256: sha256(data.bytes()),
        })
        .collect();
    let snapshots = tensors
        .into_iter()
        .enumerate()
        .map(|(i, (name, data))| pack_tensor(data, name.into(), Some(i as u64 + 1)))
        .collect();
    let bytes = BurnpackWriter::new(snapshots)
        .into_bytes()
        .unwrap()
        .to_vec();
    let object = Object {
        stage: "fixture".into(),
        size: bytes.len(),
        sha256: sha256(&bytes),
        parts: vec![Part {
            size: bytes.len(),
            sha256: sha256(&bytes),
        }],
        tensors: specs,
    };
    (object, bytes)
}

#[test]
fn tensor_views_keep_the_authenticated_object_alive_without_copying() {
    let (object, bytes) = fixture(vec![1.0, 2.0, 3.0]);
    let start = bytes.as_ptr() as usize;
    let end = start + bytes.len();
    let tensors = pollster::block_on(async {
        let mut reader = Reader(Some(bytes));
        read_tensors(&mut reader, &object).await.unwrap()
    }); // The reader, Burnpack parser and snapshots are already gone.
    for data in tensors.values() {
        let ptr = data.bytes().as_ptr() as usize;
        assert!(ptr >= start && ptr + data.bytes().len() <= end);
    }
    assert_eq!(
        tensors["first"].try_to_vec::<f32>().unwrap(),
        [1.0, 2.0, 3.0]
    );
    assert_eq!(tensors["second"].try_to_vec::<f32>().unwrap(), [4.0, 5.0]);
}

#[test]
fn shared_views_still_reject_invalid_tensor_inventory_and_values() {
    let (object, bytes) = fixture(vec![1.0, 2.0, 3.0]);
    for bad in 0..5 {
        let mut object = object.clone();
        match bad {
            0 => object.tensors[0].sha256 = "0".repeat(64),
            1 => object.tensors[0].shape = vec![1, 3],
            2 => object.tensors[0].dtype = "i32".into(),
            3 => {
                object.tensors.pop();
            }
            _ => object.tensors.push(object.tensors[0].clone()),
        }
        assert!(
            pollster::block_on(read_tensors(&mut Reader(Some(bytes.clone())), &object)).is_err()
        );
    }
    let (object, bytes) = fixture(vec![1.0, f32::NAN, 3.0]);
    assert!(pollster::block_on(read_tensors(&mut Reader(Some(bytes)), &object)).is_err());
}

fn legacy_fixture(block_size: u8) -> (Object, Vec<u8>) {
    let payload = [vec![0u8; 16], 1f32.to_le_bytes().to_vec()].concat();
    let metadata = serde_json::json!({"tensors":{"legacy.weight":{
        "dtype":{"QFloat":{"value":"Q4F","param":"F32","store":{"PackedU32":0},"level":{"Block":{"storage":[block_size,1,1,1,1],"len":1}},"mode":"Symmetric"}},
        "shape":[1,32],"data_offsets":[0,20],"param_id":1
    }}});
    let mut cbor = Vec::new();
    ciborium::into_writer(&metadata, &mut cbor).unwrap();
    let mut bytes = b"NRUB\x01\x00".to_vec();
    bytes.extend_from_slice(&(cbor.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&cbor);
    bytes.resize(bytes.len().next_multiple_of(256), 0);
    bytes.extend_from_slice(&payload);
    let part = Part {
        size: bytes.len(),
        sha256: sha256(&bytes),
    };
    let object = Object {
        stage: "legacy".into(),
        size: bytes.len(),
        sha256: sha256(&bytes),
        parts: vec![part],
        tensors: vec![TensorSpec {
            name: "legacy.weight".into(),
            shape: vec![1, 32],
            dtype: "q4f32".into(),
            sha256: sha256(&payload),
        }],
    };
    (object, bytes)
}
#[test]
fn burn_021_q4_metadata_is_normalized_without_copying_the_cdn_payload() {
    let (object, bytes) = legacy_fixture(32);
    let start = bytes.as_ptr() as usize;
    let end = start + bytes.len();
    let tensors = pollster::block_on(read_tensors(&mut Reader(Some(bytes)), &object)).unwrap();
    let data = &tensors["legacy.weight"];
    assert_eq!(data.dtype(), burn_human_inference::dtype("q4f32").unwrap());
    let ptr = data.bytes().as_ptr() as usize;
    assert!(ptr >= start && ptr + data.bytes().len() <= end);
    let (object, bytes) = legacy_fixture(16);
    assert!(pollster::block_on(read_tensors(&mut Reader(Some(bytes)), &object)).is_err());
}
