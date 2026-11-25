/* smtdif: how hard an SMT formula is, judged by its shape */
pub mod buf;
pub mod json;
pub mod sexp;
pub mod stream;
pub mod table;
pub mod tar;

use std::eprintln;
use std::fs::File;
use std::io::{self, Read};
use std::process::exit;

pub type Res<T> = Result<T, String>;

/* report a failure and leave, in the way of sysfatal */
pub fn
die(prog: &str, msg: &str)
{
	eprintln!("{}: {}", prog, msg);
	exit(1);
}

/* the input of a program: what comes after its flags, the standard input when nothing does */
pub fn
input(argv: &[String]) -> Res<Box<dyn Read>>
{
	match(argv){
		[] => Ok(Box::new(io::stdin())),
		[p] if p == "-" => Ok(Box::new(io::stdin())),
		[p] => Ok(Box::new(File::open(p).map_err(|e| format!("can't open {}: {}", p, e))?)),
		_ => Err("one input at most".to_string()),
	}
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn input_is_a_file_or_nothing()
	{
		let mut text = String::new();
		input(&["Cargo.toml".to_string()]).unwrap().read_to_string(&mut text).unwrap();
		assert_eq!(text, std::fs::read_to_string("Cargo.toml").unwrap());
		assert!(input(&["no-such-file".to_string()]).is_err());
		assert!(input(&["a".to_string(), "b".to_string()]).is_err());
	}
}
