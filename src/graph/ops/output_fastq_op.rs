use rustc_hash::FxHashMap;
use smallvec::{smallvec, SmallVec};
use std::borrow::Borrow;
use std::cell::{Cell, RefCell};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use thread_local::ThreadLocal;

use flate2::{write::GzEncoder, Compression};

use crate::graph::*;

const MEGABYTE: usize = 65536;

pub struct LocalSetBuffer {
    raw_dat: SmallVec<[Cursor<Vec<u8>>; 8]>,
    dim: usize,
}

impl LocalSetBuffer {
    pub fn new(nb: usize) -> Self {
        assert!(nb > 0);
        Self {
            raw_dat: smallvec![Cursor::new(vec![]); nb],
            dim: nb,
        }
    }

    pub fn add_read(&mut self, idx: usize, name: &[u8], seq: &[u8], qual: &[u8]) {
        let writer = &mut self.raw_dat[idx];
        writer.write_all(b"@").unwrap();
        writer.write_all(name).unwrap();
        writer.write_all(b"\n").unwrap();
        writer.write_all(seq).unwrap();
        writer.write_all(b"\n+\n").unwrap();
        writer.write_all(qual).unwrap();
        writer.write_all(b"\n").unwrap();
    }

    pub fn clear(&mut self) {
        self.raw_dat.iter_mut().for_each(|b| {
            b.get_mut().clear();
            b.set_position(0)
        });
    }

    pub fn num_bytes(&self) -> usize {
        self.raw_dat
            .iter()
            .fold(0, |acc, x| acc + x.get_ref().len())
    }
}

pub struct LocalBuffer {
    raw_dat: SmallVec<[Vec<u8>; 8]>,
    lens: SmallVec<[Vec<(usize, usize)>; 8]>,
    dim: usize,
}

impl LocalBuffer {
    pub fn new(nb: usize) -> Self {
        assert!(nb > 0);
        Self {
            raw_dat: smallvec![vec![]; nb],
            lens: smallvec![vec![]; nb],
            dim: nb,
        }
    }

    pub fn len(&self) -> usize {
        self.lens[0].len()
    }

    pub fn num_bytes(&self) -> usize {
        self.raw_dat.iter().fold(0, |acc, x| acc + x.len())
    }

    pub fn clear(&mut self) {
        for i in 0..self.dim {
            self.raw_dat[i].clear();
            self.lens[i].clear();
        }
    }

    pub fn add_read(&mut self, idx: usize, name: &[u8], seq: &[u8], qual: &[u8]) {
        let ln = name.len();
        let ls = seq.len();
        let mut rd = &mut self.raw_dat[idx];
        rd.extend_from_slice(name);
        rd.extend_from_slice(seq);
        rd.extend_from_slice(qual);
        self.lens[idx].push((ln, ls))
    }

    pub fn iter(&self, dim_idx: usize) -> LocalBufferIter {
        LocalBufferIter {
            lb: &self,
            dim_idx,
            idx: 0,
            offset: 0,
        }
    }
}

struct LocalBufferIter<'a> {
    pub lb: &'a LocalBuffer,
    pub dim_idx: usize,
    pub idx: usize,
    pub offset: usize,
}

impl<'a> Iterator for LocalBufferIter<'a> {
    type Item = (&'a [u8], &'a [u8], &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.lb.len() {
            let (nl, sl) = self.lb.lens[self.dim_idx][self.idx];
            let res = (
                &self.lb.raw_dat[self.dim_idx][self.offset..self.offset + nl],
                &self.lb.raw_dat[self.dim_idx][self.offset + nl..self.offset + nl + sl],
                &self.lb.raw_dat[self.dim_idx][self.offset + nl + sl..self.offset + nl + sl + sl],
            );
            self.offset += nl + sl + sl;
            self.idx += 1;
            return Some(res);
        } else {
            return None;
        }
    }
}

fn init_writers(file_names: &[PathBuf]) -> std::io::Result<Vec<Box<dyn Write + Send>>> {
    let mut file_writers = vec![]; //self.file_writers.lock().unwrap();

    for file_path in file_names {
        // need to create the output file
        if let Some(parent) = std::path::Path::new(file_path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let writer: Box<dyn Write + Send> = if file_path.ends_with(".gz") {
            Box::new(BufWriter::new(GzEncoder::new(
                File::create(file_path)?,
                Compression::default(),
            )))
        } else {
            Box::new(BufWriter::new(File::create(file_path)?))
        };

        file_writers.push(writer);
    }
    Ok(file_writers)
}

pub struct OutputFastqFileOp {
    required_names: Vec<LabelOrAttr>,
    file_exprs: Vec<Expr>,
    file_paths: Vec<std::path::PathBuf>,
    file_writers: Mutex<Vec<Box<dyn Write + Send>>>, //FxHashMap<Vec<u8>, Arc<Mutex<dyn Write + Send>>>>,
    buffer: ThreadLocal<RefCell<LocalSetBuffer>>,
}

impl OutputFastqFileOp {
    const NAME: &'static str = "OutputFastqFileOp";

    /// Output reads (read 1 only) to a file whose path is specified by an expression.
    pub fn from_file(file_expr: impl Into<Expr>) -> Self {
        let file_expr = file_expr.into();
        let read = Read::new();
        let file_name = file_expr.eval_bytes(&read, false);
        let file_path = PathBuf::from(std::str::from_utf8(file_name.unwrap().borrow()).unwrap());

        let b = LocalSetBuffer::new(1);
        let c = RefCell::new(b);
        let t = ThreadLocal::new();
        t.get_or(|| c);

        let file_paths = vec![file_path];
        let fwriters = Mutex::new(init_writers(&file_paths).expect("couldn't initialize writers"));
        Self {
            required_names: file_expr.required_names(),
            file_exprs: vec![file_expr],
            file_paths,
            file_writers: fwriters,
            buffer: t,
        }
    }

    /// Output reads to separate files whose paths are specified by expressions.
    pub fn from_files<E: Into<Expr>>(file_exprs: impl IntoIterator<Item = E>) -> Self {
        let file_exprs = file_exprs.into_iter().map(|e| e.into()).collect::<Vec<_>>();

        let read = Read::new();
        let file_paths = file_exprs
            .iter()
            .map(|e| {
                let fname = e.eval_bytes(&read, false);
                let fstr = String::from(std::str::from_utf8(fname.unwrap().borrow()).unwrap());
                std::path::PathBuf::from(fstr)
            })
            .collect::<Vec<_>>();

        let required_names = file_exprs
            .iter()
            .flat_map(|e| e.required_names().into_iter())
            .collect::<Vec<_>>();

        let fwriters = Mutex::new(init_writers(&file_paths).expect("couldn't initialize writers"));
        Self {
            required_names,
            file_exprs,
            file_paths,
            file_writers: fwriters,
            buffer: ThreadLocal::<RefCell<LocalSetBuffer>>::new(),
        }
    }

    // get the corresponding file writer for each read first so writing to different files can be parallelized
    /*
    fn get_writer(&self, file_name: &[u8]) -> std::io::Result<Arc<Mutex<dyn Write + Send>>> {
        use std::collections::hash_map::Entry::*;
        let mut file_writers = self.file_writers.lock().unwrap();

        match file_writers.entry(file_name.to_owned()) {
            Occupied(e) => Ok(Arc::clone(e.get())),
            Vacant(e) => {
                // need to create the output file
                let file_path = std::str::from_utf8(file_name).unwrap();

                if let Some(parent) = std::path::Path::new(file_path).parent() {
                    std::fs::create_dir_all(parent)?;
                }

                let writer: Arc<Mutex<dyn Write + Send>> = if file_path.ends_with(".gz") {
                    Arc::new(Mutex::new(BufWriter::new(GzEncoder::new(
                        File::create(file_path)?,
                        Compression::default(),
                    ))))
                } else {
                    Arc::new(Mutex::new(BufWriter::new(File::create(file_path)?)))
                };

                Ok(Arc::clone(e.insert(writer)))
            }
        }
    }
    */
}

impl<T: Trace> GraphNode<T> for OutputFastqFileOp {
    fn run_inner(&self, read: Read) -> Result<(Option<Read>, bool)> {
        let mut local_buffer = self
            .buffer
            .get_or(|| RefCell::new(LocalSetBuffer::new(self.file_paths.len())))
            .borrow_mut();

        for i in 0..self.file_paths.len() {
            // get the i-th record in the read
            let record = read.to_fastq((i + 1) as _).map_err(|e| Error::NameError {
                source: e,
                read: read.clone(),
                context: Self::NAME,
            })?;
            local_buffer.add_read(i, record.0, record.1, record.2);
        }
        // if we exceeded the buffer size, then acquire the lock and write the output
        if local_buffer.num_bytes() >= MEGABYTE {
            let mut writer_lock = self.file_writers.lock().unwrap();
            for i in 0..self.file_paths.len() {
                local_buffer.raw_dat[i].set_position(0);
                std::io::copy(&mut local_buffer.raw_dat[i], &mut writer_lock[i]);
            }
            local_buffer.clear();
        }
        Ok((Some(read), false))
    }

    fn required_names(&self) -> &[LabelOrAttr] {
        &self.required_names
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn finish(&self) -> Result<bool> {
        if let Some(rc) = self.buffer.get() {
            let mut local_buffer = rc.borrow_mut();
            let mut writer_lock = self.file_writers.lock().unwrap();
            for i in 0..self.file_paths.len() {
                local_buffer.raw_dat[i].set_position(0);
                std::io::copy(&mut local_buffer.raw_dat[i], &mut writer_lock[i]);
            }
            local_buffer.clear();
        }
        Ok(true)
    }
}

pub struct OutputFastqOp<'writer> {
    writers: Vec<Mutex<Box<dyn Write + Send + 'writer>>>,
}

impl<'writer> OutputFastqOp<'writer> {
    const NAME: &'static str = "OutputFastqOp";

    /// Output reads (read 1 only) to a `Write`r.
    pub fn from_writer(writer: impl Write + Send + 'writer) -> Self {
        Self {
            writers: vec![Mutex::new(Box::new(writer))],
        }
    }

    /// Output reads to separate `Write`rs.
    pub fn from_writers<W: Write + Send + 'writer>(writers: impl IntoIterator<Item = W>) -> Self {
        Self {
            writers: writers
                .into_iter()
                .map(|w| {
                    let w: Box<dyn Write + Send + 'writer> = Box::new(w);
                    Mutex::new(w)
                })
                .collect(),
        }
    }
}

impl<'writer, T: Trace> GraphNode<T> for OutputFastqOp<'writer> {
    fn run_inner(&self, read: Read) -> Result<(Option<Read>, bool)> {
        for (i, writer) in self.writers.iter().enumerate() {
            let record = read.to_fastq((i + 1) as _).map_err(|e| Error::NameError {
                source: e,
                read: read.clone(),
                context: Self::NAME,
            })?;

            let mut writer = writer.lock().unwrap();
            write_fastq_record(&mut *writer, record);
        }

        Ok((Some(read), false))
    }

    fn required_names(&self) -> &[LabelOrAttr] {
        &[]
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn finish(&self) -> Result<bool> {
        Ok(true)
    }
}

pub fn write_fastq_record(
    writer: &mut (dyn Write + std::marker::Send),
    record: (&[u8], &[u8], &[u8]),
) {
    writer.write_all(b"@").unwrap();
    writer.write_all(&record.0).unwrap();
    writer.write_all(b"\n").unwrap();
    writer.write_all(&record.1).unwrap();
    writer.write_all(b"\n+\n").unwrap();
    writer.write_all(&record.2).unwrap();
    writer.write_all(b"\n").unwrap();
}

pub fn record_size(record: (&[u8], &[u8], &[u8])) -> usize {
    let (source, read, context) = record;
    core::mem::size_of_val(source) + core::mem::size_of_val(read) + core::mem::size_of_val(context)
}
