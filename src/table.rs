/* table: text in rows, with a header naming the columns, tabs between fields */
use std::io::{self, BufRead, BufReader, Read, Write};
use crate::Res;

pub struct Table {
	pub cols: Vec<String>,
	pub rows: Vec<Vec<String>>,
}

fn
fields(line: &str) -> Vec<String>
{
	line.split('\t').map(String::from).collect()
}

/* where the column of that name stands */
pub fn
col(cols: &[String], name: &str) -> Res<usize>
{
	cols.iter().position(|c| c == name).ok_or(format!("no column {}", name))
}

impl Table {
	pub fn
	read<R: Read>(r: R) -> Res<Table>
	{
		let mut r = BufReader::new(r);
		let mut line = String::new();
		if(r.read_line(&mut line).map_err(|e| e.to_string())? == 0){
			return Err("table: empty".to_string());
		}
		let cols = fields(line.trim_end_matches(['\r', '\n']));
		let mut rows = Vec::new();
		loop {
			line.clear();
			if(r.read_line(&mut line).map_err(|e| e.to_string())? == 0){
				break;
			}
			let text = line.trim_end_matches(['\r', '\n']);
			if(text.is_empty()){
				continue;
			}
			let f = fields(text);
			if(f.len() != cols.len()){
				return Err(format!("table: row {} has {} fields, the header {}", rows.len() + 1, f.len(), cols.len()));
			}
			rows.push(f);
		}
		Ok(Table { cols, rows })
	}

	pub fn
	col(&self, name: &str) -> Res<usize>
	{
		col(&self.cols, name)
	}

	pub fn
	write<W: Write>(&self, w: &mut W) -> io::Result<()>
	{
		writeln!(w, "{}", self.cols.join("\t"))?;
		for r in &self.rows{
			writeln!(w, "{}", r.join("\t"))?;
		}
		Ok(())
	}
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn round_trip()
	{
		let text = "id\tlogic\tn\na\tQF_BV\t1\n\nb\tQF_BV\t2\r\n";
		let t = Table::read(text.as_bytes()).unwrap();
		assert_eq!(t.cols, ["id", "logic", "n"]);
		assert_eq!(t.rows, [["a", "QF_BV", "1"], ["b", "QF_BV", "2"]]);
		assert_eq!(t.col("logic"), Ok(1));
		assert!(t.col("nope").is_err());
		let mut back = Vec::new();
		t.write(&mut back).unwrap();
		assert_eq!(String::from_utf8(back).unwrap(), "id\tlogic\tn\na\tQF_BV\t1\nb\tQF_BV\t2\n");
	}

	#[test]
	fn ragged_rows()
	{
		assert!(Table::read("a\tb\n1\n".as_bytes()).is_err());
		assert!(Table::read("".as_bytes()).is_err());
	}
}
