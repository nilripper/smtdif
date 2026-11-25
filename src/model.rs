/* model: what every learner offers, a prediction for a row of features */
use crate::stat::median;

/* the rows of one task, as the learners take them: features scaled, the target apart */
pub struct Mat {
	pub n: usize,
	pub d: usize,
	/* the classes; 0 when the task is regression */
	pub k: usize,
	/* n rows of d features */
	pub x: Vec<f64>,
	/* log10 of the time */
	pub y: Vec<f64>,
	pub c: Vec<usize>,
}

impl Mat {
	pub fn
	row(&self, i: usize) -> &[f64]
	{
		&self.x[i * self.d..i * self.d + self.d]
	}
}

pub trait Model {
	/* the estimate, or the class as a number */
	fn predict(&self, x: &[f64]) -> f64;
}

/* a learner family: the settings cross validation tries, and the way to fit one */
pub struct Fam {
	pub name: &'static str,
	pub names: &'static [&'static str],
	/* the settings tried at regression, each a vector in the order of names */
	pub gridr: Vec<Vec<f64>>,
	pub gridc: Vec<Vec<f64>>,
	pub fit: fn(&[f64], &Mat) -> Box<dyn Model>,
	/* the fit keeps a square matrix over the training rows */
	pub square: bool,
}

impl Fam {
	/* a setting as text, as "k=5" */
	pub fn
	show(&self, p: &[f64]) -> String
	{
		let mut s = Vec::new();
		for (n, v) in self.names.iter().zip(p){
			s.push(format!("{}={}", n, v));
		}
		s.join(" ")
	}
}

/* the guess that needs no features: one and the same value for every row */
struct Same(f64);

impl Model for Same {
	fn
	predict(&self, _: &[f64]) -> f64
	{
		self.0
	}
}

/* the median of the training times */
pub fn
medianfit(_: &[f64], m: &Mat) -> Box<dyn Model>
{
	Box::new(Same(median(&m.y)))
}

/* the most common class of the training rows, the lowest numbered when tied */
pub fn
majorityfit(_: &[f64], m: &Mat) -> Box<dyn Model>
{
	let mut n = vec![0usize; m.k];
	for c in &m.c{
		n[*c] += 1;
	}
	let best = (0..m.k).max_by_key(|c| (n[*c], std::cmp::Reverse(*c))).unwrap_or(0);
	Box::new(Same(best as f64))
}

/* cross products of the settings of each name: [a, b] over two names gives aa, ab, ba, bb */
pub fn
grid(settings: &[&[f64]]) -> Vec<Vec<f64>>
{
	let mut g = vec![Vec::new()];
	for s in settings{
		let mut next = Vec::new();
		for p in &g{
			for v in *s{
				let mut q = p.clone();
				q.push(*v);
				next.push(q);
			}
		}
		g = next;
	}
	g
}

/* the families in the order the tables show them */
pub fn
families() -> Vec<Fam> {
	vec![
		Fam {
			name: "median",
			names: &[],
			gridr: vec![vec![]],
			gridc: vec![],
			fit: medianfit,
			square: false,
		},
		Fam {
			name: "majority",
			names: &[],
			gridr: vec![],
			gridc: vec![vec![]],
			fit: majorityfit,
			square: false,
		},
		Fam {
			name: "ridge",
			names: &["lambda"],
			gridr: grid(&[&[1e-4, 1e-3, 1e-2, 1e-1, 1.0]]),
			gridc: vec![],
			fit: crate::lin::ridge,
			square: false,
		},
		Fam {
			name: "logistic",
			names: &["lambda"],
			gridr: vec![],
			gridc: grid(&[&[1e-4, 1e-3, 1e-2, 1e-1, 1.0]]),
			fit: crate::lin::logisticfit,
			square: false,
		},
		Fam {
			name: "knn",
			names: &["k"],
			gridr: grid(&[&[1.0, 3.0, 5.0, 11.0, 21.0, 41.0]]),
			gridc: grid(&[&[1.0, 3.0, 5.0, 11.0, 21.0, 41.0]]),
			fit: crate::knn::knn,
			square: false,
		},
		Fam {
			name: "bayes",
			names: &[],
			gridr: vec![],
			gridc: vec![vec![]],
			fit: crate::bayes::bayes,
			square: false,
		},
		Fam {
			name: "tree",
			names: &["minleaf"],
			gridr: grid(&[&[2.0, 10.0, 30.0, 100.0]]),
			gridc: grid(&[&[2.0, 10.0, 30.0, 100.0]]),
			fit: crate::tree::treefit,
			square: false,
		},
		Fam {
			name: "forest",
			names: &["minleaf"],
			gridr: grid(&[&[1.0, 5.0, 15.0]]),
			gridc: grid(&[&[1.0, 5.0, 15.0]]),
			fit: crate::forest::forest,
			square: false,
		},
		Fam {
			name: "boost",
			names: &["depth"],
			gridr: grid(&[&[2.0, 3.0, 5.0]]),
			gridc: grid(&[&[2.0, 3.0, 5.0]]),
			fit: crate::forest::boost,
			square: false,
		},
		Fam {
			name: "svm",
			names: &["C", "gamma", "epsilon"],
			gridr: grid(&[&[1.0, 10.0, 100.0], &[0.5, 1.0, 2.0], &[0.05, 0.1]]),
			gridc: grid(&[&[1.0, 10.0, 100.0], &[0.5, 1.0, 2.0]]),
			fit: crate::svm::svm,
			square: true,
		},
	]
}

#[cfg(test)]
mod test {
	use super::*;

	fn
	mat() -> Mat
	{
		Mat {
			n: 4,
			d: 1,
			k: 3,
			x: vec![0.0, 1.0, 2.0, 3.0],
			y: vec![1.0, 9.0, 2.0, 9.0],
			c: vec![2, 1, 1, 0],
		}
	}

	#[test]
	fn guesses()
	{
		let m = mat();
		let p: Vec<f64> = (0..4).map(|i| medianfit(&[], &m).predict(m.row(i))).collect();
		assert_eq!(p, [5.5; 4]);
		let p: Vec<f64> = (0..4).map(|i| majorityfit(&[], &m).predict(m.row(i))).collect();
		assert_eq!(p, [1.0; 4]);
	}

	#[test]
	fn grids_and_names()
	{
		assert_eq!(grid(&[&[1.0, 2.0], &[3.0, 4.0, 5.0]]),
			[[1.0, 3.0], [1.0, 4.0], [1.0, 5.0], [2.0, 3.0], [2.0, 4.0], [2.0, 5.0]]);
		assert_eq!(grid(&[]), [Vec::<f64>::new()]);
		for f in families(){
			/* a setting names a value for each of the names, or for those its task uses */
			for p in f.gridr.iter().chain(&f.gridc){
				assert!(p.len() <= f.names.len() && p.len() > 0 || f.names.is_empty(), "{}", f.name);
			}
			assert!(!f.gridr.is_empty() || !f.gridc.is_empty(), "{}", f.name);
		}
		let f = &families()[9];
		assert_eq!(f.name, "svm");
		assert_eq!((f.gridr.len(), f.gridc.len()), (18, 9));
		assert_eq!(f.show(&f.gridr[1]), "C=1 gamma=0.5 epsilon=0.1");
	}
}
