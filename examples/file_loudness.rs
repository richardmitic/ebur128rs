extern crate audrey;
extern crate ebur128rs;

use std::env;

pub fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: ./file_loudness <wav file>");
        return;
    }
    let mut reader = audrey::read::open(args[1].clone()).unwrap();
    let channels = reader.description().channel_count() as usize;
    let samples = reader
        .samples::<f64>()
        .flat_map(|s| s)
        .collect::<Vec<f64>>();
    let mut state = ebur128rs::State::new(48000., channels, false);

    for chunk in samples.as_slice().chunks_exact(9600) {
        let result = state.process(chunk);
        if result.is_err() {
            eprintln!("{:?}", result);
            break;
        } else {
            let ml = state
                .momentary_loudness(ebur128rs::GatingType::Absolute)
                .unwrap();
            let stl = state
                .short_term_loudness(ebur128rs::GatingType::Absolute)
                .unwrap();
            println!("momentary:{:?} short term:{:?}", ml, stl);
        }
    }

    let ila = state.integrated_loudness(ebur128rs::GatingType::Absolute);
    let ilr = state.integrated_loudness(ebur128rs::GatingType::Relative);
    println!("integrated absolute:{:?}", ila);
    println!("integrated relative:{:?}", ilr);
}
