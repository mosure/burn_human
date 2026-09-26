use anyhow::{Result, ensure};
use burn::backend::Wgpu;
use burn_gem::{
    pipeline::{Pipeline, PipelineArtifacts},
    validation,
};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 4,
        "gem-validate SUITE_JSON IMAGE REFERENCE_JSON OUTPUT_JSON"
    );
    pollster::block_on(async {
        let artifacts = PipelineArtifacts::from_location(&args[0]).await?;
        let model = Pipeline::<Wgpu>::load(&artifacts, &Default::default(), |s, i, n| {
            if i == n {
                println!("{s} loaded");
            }
        })
        .await?;
        let report =
            validation::validate(&model, &std::fs::read(&args[1])?, &std::fs::read(&args[2])?)
                .await?;
        std::fs::write(&args[3], serde_json::to_vec_pretty(&report)?)?;
        println!("{report}");
        ensure!(report["passed"] == true, "GEM-X parity failed");
        Ok(())
    })
}
