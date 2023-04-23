use antisequence::{fastq::*, iter::*, read::*};

fn main() {
    iter_fastq("example_data/simple.fastq", 256)
        .cut("", "seq.* -> seq.a, seq.b", LeftEnd(3))
        .for_each("", |read| println!("{}", read))
        .trim("", ["seq.a"])
        .for_each("", |read| println!("{}", read))
        .collect_fastq("", "example_output/simple.fastq")
        .run(1);
}
