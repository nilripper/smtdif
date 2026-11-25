/* knn: the k nearest training rows answer, by their distance in the features */
use crate::model::{Mat, Model};
use crate::stat::argmax;

struct Knn {
	k: usize,
	m: Mat,
}

impl Model for Knn {
	fn
	predict(&self, x: &[f64]) -> f64
	{
		let (m, k) = (&self.m, self.k.min(self.m.n));
		let dist: Vec<f64> = (0..m.n)
			.map(|i| m.row(i).iter().zip(x).map(|(a, b)| (a - b) * (a - b)).sum()).collect();
		/* the k lowest distances, by a partial sort; then in their order */
		let mut near: Vec<usize> = (0..m.n).collect();
		let mut first: Vec<usize> = if(k < m.n){
			let (low, _, _) = near.select_nth_unstable_by(k, |a, b| dist[*a].total_cmp(&dist[*b]));
			low.to_vec()
		}else{
			near
		};
		first.sort_unstable_by(|a, b| dist[*a].total_cmp(&dist[*b]));
		if(m.k == 0){
			/* the mean of their targets */
			return first.iter().map(|i| m.y[*i]).sum::<f64>() / k as f64;
		}
		/* a vote each, the nearer the heavier */
		let mut votes = vec![0.0; m.k];
		for i in first.iter(){
			votes[m.c[*i]] += 1.0 / (1.0 + dist[*i]);
		}
		argmax(&votes) as f64
	}
}

pub fn
knn(p: &[f64], m: &Mat) -> Box<dyn Model>
{
	Box::new(Knn { k: (p[0] as usize).max(1), m: Mat { n: m.n, d: m.d, k: m.k, x: m.x.clone(), y: m.y.clone(), c: m.c.clone() } })
}

#[cfg(test)]
mod test {
	use super::*;

	fn
	close(a: f64, b: f64)
	{
		assert!((a - b).abs() < 1e-9, "{} is not {}", a, b);
	}

	#[test]
	fn nearest()
	{
		/* the target is the first feature: a neighbor of k one gives it back */
		let mut m = Mat { n: 40, d: 2, k: 0, x: Vec::new(), y: Vec::new(), c: Vec::new() };
		for i in 0..m.n{
			let a = i as f64 * 0.5;
			m.x.extend_from_slice(&[a, 100.0 - a]);
			m.y.push(a);
			m.c.push(if a < 10.0{ 0 } else { 1 });
		}
		for i in 0..m.n{
			close(knn(&[1.0], &m).predict(m.row(i)), m.y[i]);
		}
		close(knn(&[3.0], &m).predict(&[4.0, 96.0]), 4.0);
		close(knn(&[41.0], &m).predict(&[4.0, 96.0]), m.y.iter().sum::<f64>() / 40.0);
	}

	#[test]
	fn voting()
	{
		let m = Mat {
			n: 6,
			d: 1,
			k: 3,
			x: vec![0.0, 0.1, 0.2, 9.0, 9.1, 9.2],
			y: vec![0.0; 6],
			c: vec![2, 2, 0, 1, 1, 1],
		};
		assert_eq!(knn(&[1.0], &m).predict(&[0.0]), 2.0);
		assert_eq!(knn(&[2.0], &m).predict(&[0.0]), 2.0);
		assert_eq!(knn(&[3.0], &m).predict(&[0.0]), 2.0);
		assert_eq!(knn(&[5.0], &m).predict(&[8.0]), 1.0);
		/* k past the rows counts them all, and a distant cluster still cannot outvote a near one */
		assert_eq!(knn(&[100.0], &m).predict(&[0.0]), 2.0);
	}
}
