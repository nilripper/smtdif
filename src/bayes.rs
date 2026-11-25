/* bayes: a gaussian per class and feature, the features taken as independent of one another */
use crate::model::{Mat, Model};
use crate::stat::argmax;

struct Bayes {
	d: usize,
	/* for each class: the log of its prior less half the log of the variances' product,
	 * then d means, then d inverse variances */
	par: Vec<f64>,
}

impl Model for Bayes {
	fn
	predict(&self, x: &[f64]) -> f64
	{
		let stride = 1 + 2 * self.d;
		let score: Vec<f64> = (0..self.par.len() / stride).map(|c| {
			let p = &self.par[c * stride..(c + 1) * stride];
			let l = x.iter().enumerate()
				.map(|(j, v)| -p[1 + self.d + j] * (v - p[1 + j]) * (v - p[1 + j]) / 2.0)
				.sum::<f64>();
			p[0] + l
		}).collect();
		argmax(&score) as f64
	}
}

pub fn
bayes(_: &[f64], m: &Mat) -> Box<dyn Model>
{
	/* the means and the variances of each class, with the largest variance fixing the floor */
	let (mut mu, mut var, mut top) = (vec![0.0; m.k * m.d], vec![0.0; m.k * m.d], 0.0f64);
	let mut n = vec![0usize; m.k];
	for i in 0..m.n{
		n[m.c[i]] += 1;
		for j in 0..m.d{
			mu[m.c[i] * m.d + j] += m.x[i * m.d + j];
		}
	}
	for c in 0..m.k{
		for j in 0..m.d{
			mu[c * m.d + j] /= n[c].max(1) as f64;
		}
	}
	for i in 0..m.n{
		for j in 0..m.d{
			let e = m.x[i * m.d + j] - mu[m.c[i] * m.d + j];
			var[m.c[i] * m.d + j] += e * e;
		}
	}
	for c in 0..m.k{
		for j in 0..m.d{
			var[c * m.d + j] /= n[c].max(1) as f64;
			top = top.max(var[c * m.d + j]);
		}
	}
	let floor = top * 1e-9;
	let mut par = Vec::with_capacity(m.k * (1 + 2 * m.d));
	for c in 0..m.k{
		par.push((n[c].max(1) as f64 / m.n as f64).ln()
			- var[c * m.d..(c + 1) * m.d].iter()
				.map(|v| ((v + floor) * 2.0 * std::f64::consts::PI).ln() / 2.0).sum::<f64>());
		par.extend_from_slice(&mu[c * m.d..(c + 1) * m.d]);
		par.extend(var[c * m.d..(c + 1) * m.d].iter().map(|v| 1.0 / (v + floor)));
	}
	Box::new(Bayes { d: m.d, par })
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn two_clouds()
	{
		/* two well separated clouds, the second of a far wider spread */
		let mut m = Mat { n: 0, d: 2, k: 2, x: Vec::new(), y: Vec::new(), c: Vec::new() };
		for i in 0..400{
			let (c, t) = (i % 2, (i / 2) as f64);
			let (a, b) = if(c == 0){ (t / 500.0, -t / 400.0) } else { (8.0 + t / 20.0, 8.0 - t / 25.0) };
			m.x.extend_from_slice(&[a, b]);
			m.y.push(0.0);
			m.c.push(c);
			m.n += 1;
		}
		let b = bayes(&[], &m);
		for i in 0..m.n{
			assert_eq!(b.predict(m.row(i)) as usize, m.c[i], "row {}", i);
		}
		assert_eq!(b.predict(&[0.2, 0.1]), 0.0);
		assert_eq!(b.predict(&[8.2, 7.9]), 1.0);
	}

	#[test]
	fn a_feature_that_never_varies()
	{
		let m = Mat {
			n: 4,
			d: 2,
			k: 2,
			x: vec![0.0, 5.0, 0.5, 5.0, 9.0, 5.0, 9.5, 5.0],
			y: vec![0.0; 4],
			c: vec![0, 0, 1, 1],
		};
		assert!(bayes(&[], &m).predict(&[0.0, 5.0]).is_finite());
	}
}
