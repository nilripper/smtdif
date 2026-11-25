/* split: who goes into which part, at random or keeping every family whole */
use std::cmp::Reverse;
use crate::rng::Rand;

/* deal n rows into parts of the given shares. under groups, a group lands whole in one part */
pub fn
deal(groups: Option<&[usize]>, n: usize, shares: &[f64], rand: &mut Rand) -> Vec<Vec<usize>>
{
	let p = shares.len();
	let mut parts: Vec<Vec<usize>> = vec![Vec::new(); p];
	match(groups){
		None => {
			let mut all: Vec<usize> = (0..n).collect();
			rand.shuffle(&mut all);
			let (mut at, mut sum) = (0, 0.0);
			for i in 0..p{
				sum += shares[i];
				let end = if(i == p - 1){ n } else { (sum * n as f64).round() as usize };
				parts[i] = all[at..end].to_vec();
				at = end;
			}
		}
		Some(g) => {
			let mut members: Vec<Vec<usize>> = vec![Vec::new(); g.iter().copied().max().map(|m| m + 1).unwrap_or(0)];
			for i in 0..n{
				members[g[i]].push(i);
			}
			/* the largest families choose first, so that each finds a part with room */
			let mut order: Vec<usize> = (0..members.len()).collect();
			rand.shuffle(&mut order);
			order.sort_by_key(|a| Reverse(members[*a].len()));
			let room: Vec<f64> = shares.iter().map(|s| s * n as f64).collect();
			let mut size = vec![0usize; p];
			for a in order{
				let left = |i: usize| room[i] - size[i] as f64;
				/* the part with the most room left, the first of those as good */
				let to = (0..p).min_by(|a, b| left(*b).total_cmp(&left(*a))).unwrap_or(0);
				parts[to].extend_from_slice(&members[a]);
				size[to] += members[a].len();
			}
		}
	}
	for part in &mut parts{
		part.sort_unstable();
	}
	parts
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn at_random()
	{
		let mut r = Rand::new(3);
		let p = deal(None, 100, &[0.7, 0.15, 0.15], &mut r);
		assert_eq!(p.iter().map(|x| x.len()).collect::<Vec<_>>(), [70, 15, 15]);
		let mut all: Vec<usize> = p.concat();
		all.sort_unstable();
		assert_eq!(all, (0..100).collect::<Vec<usize>>());
		assert_eq!(deal(None, 100, &[0.7, 0.15, 0.15], &mut Rand::new(3)), p);
	}

	#[test]
	fn by_family()
	{
		/* families 0 and 1 are too large for any part but the first */
		let mut groups = vec![0usize; 100];
		groups[40..70].fill(1);
		groups[70..].fill(2);
		groups[90..].fill(3);
		let p = deal(Some(&groups), 100, &[0.7, 0.15, 0.15], &mut Rand::new(3));
		assert_eq!(p.iter().map(|x| x.len()).collect::<Vec<_>>(), [70, 20, 10]);
		/* a family may share a part with another, but never straddle two */
		let mut whereof: Vec<Option<usize>> = vec![None; 4];
		for (i, part) in p.iter().enumerate(){
			for row in part{
				let g = groups[*row];
				let was = whereof[g];
				whereof[g] = Some(i);
				assert_eq!(was.map_or(i, |w| w), i, "family {} straddles two parts", g);
			}
		}
		assert_eq!(whereof, [Some(0), Some(0), Some(1), Some(2)]);
		/* four folds of one family each */
		let f = deal(Some(&groups), 100, &[0.25, 0.25, 0.25, 0.25], &mut Rand::new(4));
		assert_eq!(f.iter().map(|x| x.len()).collect::<Vec<_>>(), [40, 30, 20, 10]);
	}
}
