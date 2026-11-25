/* data: the dataset as numbers, with the targets read off it */
use crate::table::Table;
use crate::Res;

/* the classes, in the order smtdata and smtfit number them */
pub static CLASSES: [&str; 3] = ["sat", "unsat", "timeout"];

/* the columns of the dataset that are not features */
static META: [&str; 7] = ["id", "logic", "family", "status", "result", "wall", "cpu"];

pub struct Data {
	pub n: usize,
	pub d: usize,
	pub id: Vec<String>,
	pub logic: Vec<String>,
	/* the family of each instance, as its number into families */
	pub group: Vec<usize>,
	pub families: Vec<String>,
	/* n rows of d features */
	pub x: Vec<f64>,
	/* log10 of the seconds taken, a timeout counted at twice the limit */
	pub y: Vec<f64>,
	pub solved: Vec<bool>,
	pub c: Vec<usize>,
	pub feats: Vec<String>,
}

impl Data {
	/* a table as smtdata writes them; status, the declared answer, is left out with the rest of the metadata */
	pub fn
	read(t: &Table, limit: f64) -> Res<Data>
	{
		let ci = t.col("id")?;
		let cl = t.col("logic")?;
		let cf = t.col("family")?;
		let cr = t.col("result")?;
		let cw = t.col("wall")?;
		let (mut feats, mut fc) = (Vec::new(), Vec::new());
		for (k, c) in t.cols.iter().enumerate(){
			if !META.contains(&c.as_str()){
				feats.push(c.clone());
				fc.push(k);
			}
		}
		let (n, d) = (t.rows.len(), fc.len());
		let mut data = Data {
			n,
			d,
			id: Vec::with_capacity(n),
			logic: Vec::with_capacity(n),
			group: Vec::with_capacity(n),
			families: Vec::new(),
			x: Vec::with_capacity(n * d),
			y: Vec::with_capacity(n),
			solved: Vec::with_capacity(n),
			c: Vec::with_capacity(n),
			feats,
		};
		for r in &t.rows{
			let bad = |what: &str| format!("{}: {}", r[ci], what);
			data.id.push(r[ci].clone());
			data.logic.push(r[cl].clone());
			match data.families.iter().position(|f| *f == r[cf]){
				Some(g) => data.group.push(g),
				None => {
					data.group.push(data.families.len());
					data.families.push(r[cf].clone());
				}
			}
			for c in &fc{
				data.x.push(r[*c].parse().map_err(|_| bad(&format!("{} is not a number", t.cols[*c])))?);
			}
			let wall = r[cw].parse().map_err(|_| bad("wall is not a number"))?;
			let c = CLASSES.iter().position(|s| *s == r[cr]).ok_or_else(|| bad("no such result"))?;
			data.c.push(c);
			data.solved.push(c != 2);
			data.y.push(if(c != 2){ wall } else { 2.0 * limit }.log10());
		}
		Ok(data)
	}

	pub fn
	row(&self, i: usize) -> &[f64]
	{
		&self.x[i * self.d..i * self.d + self.d]
	}
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn reading()
	{
		let text = "id\tlogic\tfamily\tstatus\tresult\twall\tcpu\tbytes\tdepth\n\
			a/QF_BV/f/x.smt2\tQF_BV\tf\tsat\tsat\t10\t9\t100\t3\n\
			b/QF_BV/f/y.smt2\tQF_BV\tf\tunsat\ttimeout\t1201\t1200\t200\t5\n\
			c/QF_ABV/g/z.smt2\tQF_ABV\tg\tunknown\tunsat\t0.5\t0.4\t300\t1\n";
		let t = Table::read(text.as_bytes()).unwrap();
		let d = Data::read(&t, 1200.0).unwrap();
		assert_eq!((d.n, d.d), (3, 2));
		assert_eq!(d.feats, ["bytes", "depth"]);
		assert_eq!(d.families, ["f", "g"]);
		assert_eq!(d.group, [0, 0, 1]);
		assert_eq!(d.c, [0, 2, 1]);
		assert_eq!(d.solved, [true, false, true]);
		assert_eq!(d.x, [100.0, 3.0, 200.0, 5.0, 300.0, 1.0]);
		assert_eq!(d.row(1), [200.0, 5.0]);
		assert!((d.y[0] - 1.0).abs() < 1e-12);
		assert!((d.y[1] - 2400.0f64.log10()).abs() < 1e-12);
		assert!((d.y[2] - 0.5f64.log10()).abs() < 1e-12);
	}

	#[test]
	fn refusals()
	{
		let t = Table::read("id\tlogic\tfamily\tstatus\tresult\twall\tcpu\tdepth\na\tQF_BV\tf\tsat\tsat\t10\t9\t\n".as_bytes()).unwrap();
		assert!(Data::read(&t, 1200.0).is_err());
		let t = Table::read("id\tlogic\tfamily\tstatus\tresult\twall\tcpu\tdepth\na\tQF_BV\tf\tsat\tmaybe\t10\t9\t3\n".as_bytes()).unwrap();
		assert!(Data::read(&t, 1200.0).is_err());
		let t = Table::read("id\tdepth\na\t3\n".as_bytes()).unwrap();
		assert!(Data::read(&t, 1200.0).is_err());
	}
}
