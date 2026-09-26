use burn_gem::{
    camera::{Camera, Crop},
    image_input,
};

#[test]
fn camera_and_crop_reject_invalid_inputs() {
    let camera = Camera {
        focal: [800.0; 2],
        center: [320.0, 240.0],
    };
    let crop = Crop {
        center: [400.0, 160.0],
        size: 400.0,
    };
    assert_eq!(crop.condition(camera), [0.1, -0.1, 0.5]);
    assert!(
        Camera {
            focal: [f32::NAN, 1.0],
            ..camera
        }
        .validate()
        .is_err()
    );
    assert!(Crop { size: 0.0, ..crop }.validate().is_err());
    assert!(
        Crop {
            center: [f32::INFINITY, 0.0],
            ..crop
        }
        .validate()
        .is_err()
    );
}

#[test]
fn rgb_crop_preserves_channels_and_black_borders() {
    let image = image::RgbImage::from_pixel(256, 256, image::Rgb([255, 0, 0]));
    let crop = Crop {
        center: [127.5; 2],
        size: 255.0,
    };
    let data = image_input::crop_rgb(&image, crop, false).unwrap();
    assert_eq!(data.len(), 3 * 256 * 192);
    for (channel, expected) in [
        (0, (1.0 - 0.485) / 0.229),
        (1, -0.456 / 0.224),
        (2, -0.406 / 0.225),
    ] {
        assert!(
            data[channel * 256 * 192..(channel + 1) * 256 * 192]
                .iter()
                .all(|x| (x - expected).abs() < 1e-5)
        );
    }
    let black = image_input::crop_rgb(
        &image,
        Crop {
            center: [-1000.0; 2],
            ..crop
        },
        false,
    )
    .unwrap();
    assert!((black[0] + 0.485 / 0.229).abs() < 1e-6);
    assert!(image_input::decode(b"invalid PNG").is_err());
}

#[test]
fn flip_test_keeps_left_right_joint_identity() {
    let size = 77 * 64 * 48;
    let mut heatmaps = vec![0.0; size * 2];
    // Left/right arm endpoints are joint 11/39. Mirror of x=12 is x=35.
    heatmaps[11 * 64 * 48 + 32 * 48 + 12] = 0.8;
    heatmaps[size + 39 * 64 * 48 + 32 * 48 + 35] = 1.0;
    heatmaps[39 * 64 * 48 + 16 * 48 + 36] = 0.6;
    heatmaps[size + 11 * 64 * 48 + 16 * 48 + 11] = 0.8;
    let points = image_input::keypoints(
        &heatmaps,
        Crop {
            center: [200.0, 300.0],
            size: 400.0,
        },
        true,
    )
    .unwrap();
    assert_eq!(&points[11][..2], &[125.0, 300.0]);
    assert_eq!(&points[39][..2], &[275.0, 200.0]);
    assert!((points[11][2] - 0.9).abs() < 1e-6);
    heatmaps[0] = f32::NAN;
    assert!(
        image_input::keypoints(
            &heatmaps,
            Crop {
                center: [0.0; 2],
                size: 1.0
            },
            true
        )
        .is_err()
    );
}
