/* rng: splitmix64, so that a seed decides everything that looks like chance */
pub struct Rand(u64);

impl Rand {
	pub fn
	new(seed: u64) -> Rand
	{
		Rand(seed)
	}

	pub fn
	next(&mut self) -> u64
	{
		self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
		let mut z = self.0;
		z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
		z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
		z ^ (z >> 31)
	}

	/* uniform in [0, 1) */
	pub fn
	unit(&mut self) -> f64
	{
		(self.next() >> 11) as f64 / (1u64 << 53) as f64
	}

	/* uniform below n */
	pub fn
	below(&mut self, n: usize) -> usize
	{
		((self.next() as u128 * n as u128) >> 64) as usize
	}

	/* shuffle v */
	pub fn
	shuffle<T>(&mut self, v: &mut [T])
	{
		for i in (1..v.len()).rev(){
			v.swap(i, self.below(i + 1));
		}
	}
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn seeded_and_spread()
	{
		let (mut a, mut b) = (Rand::new(7), Rand::new(7));
		for _ in 0..100{
			assert_eq!(a.next(), b.next());
		}
		assert_ne!(a.next(), Rand::new(8).next());
		let mut r = Rand::new(1);
		let mut n = [0usize; 10];
		for _ in 0..10000{
			let u = r.unit();
			assert!((0.0..1.0).contains(&u));
			n[r.below(10)] += 1;
		}
		for c in n{
			assert!((800..1200).contains(&c), "{}", c);
		}
		let mut v = (0..52).collect::<Vec<usize>>();
		r.shuffle(&mut v);
		v.sort();
		assert_eq!(v, (0..52).collect::<Vec<usize>>());
	}
}
