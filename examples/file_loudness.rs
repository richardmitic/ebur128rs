extern crate audrey;
extern crate ebur128rs;
extern crate itertools;

use itertools::Itertools;
use std::env;
use std::error::Error;

pub fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: ./file_loudness <wav file>");
        return Ok(());
    }
    let mut reader = audrey::read::open(args[1].clone())?;
    let channels = reader.description().channel_count() as usize;
    let samples = reader.samples::<f64>();
    let mut state = ebur128rs::State::new(48000., channels, false);

    for chunk in samples.chunks(9600).into_iter() {
        let buffer: Vec<f64> = chunk.flatten().collect();
        let result = state.process(buffer.into_iter());
        if result.is_err() {
            eprintln!("{:?}", result);
            break;
        } else {
            let ml = state.momentary_loudness(ebur128rs::GatingType::Absolute);
            let stl = state.short_term_loudness(ebur128rs::GatingType::Absolute);
            println!("momentary:{:.2?} short term:{:.2?}", ml, stl);
        }
    }

    let ila = state.integrated_loudness(ebur128rs::GatingType::Absolute);
    let ilr = state.integrated_loudness(ebur128rs::GatingType::Relative);
    println!("integrated absolute:{:?}", ila);
    println!("integrated relative:{:?}", ilr);

    Ok(())
}
