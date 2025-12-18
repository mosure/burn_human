use burn_human::{AnnyBody, AnnyInput};
use criterion::{Criterion, criterion_group, criterion_main};

fn load_body() -> AnnyBody {
    AnnyBody::from_reference_paths(
        "assets/model/fullbody_default.safetensors",
        "assets/model/fullbody_default.meta.json",
    )
    .expect("load reference")
}

fn bench_forward(c: &mut Criterion) {
    let body = load_body();
    let cases: Vec<String> = body.case_names().map(|s| s.to_string()).collect();
    c.bench_function("forward_all_cases", |b| {
        b.iter(|| {
            for case in &cases {
                let _out = body
                    .forward(AnnyInput::case(case.as_str()))
                    .expect("forward");
            }
        })
    });
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
