/* prep: features as the models take them: those that vary, logged, centered and scaled */
use crate::data::{Data, CLASSES};
use crate::model::Mat;
use crate::stat::{mean, sd};

/* the features worth keeping: those that vary over the rows, and copy no earlier one */
pub fn
keep(d: &Data, rows: &[usize]) -> Vec<usize>
{
	let mut kept: Vec<usize> = Vec::new();
	/* the least and the most of each feature kept, to tell a copy at a glance */
	let mut range: Vec<(f64, f64)> = Vec::new();
	for j in 0..d.d{
		let at = |i: usize| d.x[i * d.d + j];
		let (mut lo, mut hi) = (f64::MAX, f64::MIN);
		for i in rows{
			lo = lo.min(at(*i));
			hi = hi.max(at(*i));
		}
		if(lo == hi){
			continue;
		}
		let copy = kept.iter().zip(&range).any(|(k, r)| {
			*r == (lo, hi) && rows.iter().all(|i| at(*i) == d.x[i * d.d + k])
		});
		if !copy{
			kept.push(j);
			range.push((lo, hi));
		}
	}
	kept
}

/* the rows of a task over the kept features, each feature as ln(1 + x) */
pub fn
mat(d: &Data, rows: &[usize], kept: &[usize], classes: bool) -> Mat
{
	let mut m = Mat {
		n: rows.len(),
		d: kept.len(),
		k: if(classes){ CLASSES.len() } else { 0 },
		x: Vec::with_capacity(rows.len() * kept.len()),
		y: Vec::with_capacity(rows.len()),
		c: Vec::with_capacity(rows.len()),
	};
	for i in rows{
		for j in kept{
			m.x.push((1.0 + d.x[i * d.d + j]).ln());
		}
		m.y.push(d.y[*i]);
		m.c.push(d.c[*i]);
	}
	m
}

/* center and scale other as train is: every feature at a mean of 0 and a deviation of 1 there */
pub fn
scale(train: &mut Mat, other: &mut Mat)
{
	for j in 0..train.d{
		let v: Vec<f64> = (0..train.n).map(|i| train.x[i * train.d + j]).collect();
		let s = sd(&v);
		if !(s > 0.0){
			continue;
		}
		let mu = mean(&v);
		for mat in [&mut *train, &mut *other]{
			for i in 0..mat.n{
				mat.x[i * mat.d + j] = (mat.x[i * mat.d + j] - mu) / s;
			}
		}
	}
}

#[cfg(test)]
mod test {
	use super::*;
	use crate::table::Table;

	fn
	data() -> Data
	{
		let text = "id\tlogic\tfamily\tstatus\tresult\twall\tcpu\ta\tb\tb2\tc\n\
			a\tL\tf\tsat\tsat\t10\t9\t1\t2\t2\t7\n\
			b\tL\tf\tsat\tsat\t100\t9\t2\t4\t4\t7\n\
			c\tL\tg\tsat\tsat\t10\t9\t3\t6\t6\t7\n\
			d\tL\tg\tsat\tsat\t100\t9\t4\t8\t8\t7\n";
		Data::read(&Table::read(text.as_bytes()).unwrap(), 1200.0).unwrap()
	}

	fn
	close(a: f64, b: f64)
	{
		assert!((a - b).abs() < 1e-12, "{} is not {}", a, b);
	}

	#[test]
	fn kept_are_those_that_vary()
	{
		let d = data();
		assert_eq!(keep(&d, &[0, 1, 2, 3]), [0, 1]);
		assert_eq!(keep(&d, &[0, 1]), [0, 1]);
		/* twice the feature is not a copy of it, and still varies over a shorter run */
		assert_eq!(keep(&d, &[0, 2]), [0, 1]);
		assert_eq!(keep(&d, &[0]), []);
		assert_eq!(d.feats, ["a", "b", "b2", "c"]);
	}

	#[test]
	fn logged_and_scaled()
	{
		let d = data();
		let kept = keep(&d, &[0, 1, 2, 3]);
		let mut a = mat(&d, &[0, 1, 2, 3], &kept, false);
		let mut b = mat(&d, &[0], &kept, true);
		assert_eq!((a.n, a.d, a.k), (4, 2, 0));
		assert_eq!((b.n, b.d, b.k), (1, 2, 3));
		assert_eq!(a.x, [2.0f64.ln(), 3.0f64.ln(), 3.0f64.ln(), 5.0f64.ln(), 4.0f64.ln(), 7.0f64.ln(), 5.0f64.ln(), 9.0f64.ln()]);
		assert_eq!(b.c, [0]);
		scale(&mut a, &mut b);
		for j in 0..a.d{
			let v: Vec<f64> = (0..a.n).map(|i| a.x[i * a.d + j]).collect();
			assert!(mean(&v).abs() < 1e-12);
			assert!((sd(&v) - 1.0).abs() < 1e-12);
		}
		/* the other rows follow the training rows' mean and deviation, not their own:
		 * the row holds features 1 and 2, against training features 1..4 and 2, 4, 6, 8 */
		let stand = |v: f64, rows: [f64; 4]| {
			let logs: Vec<f64> = rows.iter().map(|v| (1.0 + v).ln()).collect();
			((1.0 + v).ln() - mean(&logs)) / sd(&logs)
		};
		close(b.x[0], stand(1.0, [1.0, 2.0, 3.0, 4.0]));
		close(b.x[1], stand(2.0, [2.0, 4.0, 6.0, 8.0]));
		assert!(b.x[0] < 0.0 && b.x[1] < 0.0);
	}
}
