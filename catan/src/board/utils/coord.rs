use std::cmp::Ordering;
use std::fmt;

use super::topology::CoordTopology;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Type {
    Void,
    Hex,
    Path,
    Intersection,
}

/// Enum representing the position of a coord on a grid
///
/// A more detailed version of [CoordType](enum.CoordType.html)
///
/// Bellow is a representation of the different position using the first letters of the variants:
///         A
///     Z       S
/// V               V
///
/// I   L   H   R   I
///
/// A               A
///     S       Z
///         V
///
pub enum DetailedType {
    /// The empty position between a hex center and it's left path
    LVoid,
    /// The empty position between a hex center and it's right path
    RVoid,
    /// The center of a hex
    OHex,
    /// The path at the top right or bottom left of a hex
    SPath,
    /// The path at the top left or bottom right of a hex
    ZPath,
    /// The vertical path at the left or right of a hex
    IPath,
    /// The intersection found bellow a hex
    VIntersection,
    /// The intersection found above a hex
    AIntersection,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Coord {
    pub x: i8,
    pub y: i8,
}

impl Ord for Coord {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.y.cmp(&other.y) {
            Ordering::Equal => self.x.cmp(&other.x),
            v => v,
        }
    }
}

impl PartialOrd for Coord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.x, self.y)
    }
}

impl fmt::Debug for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Coord({},{})", self.x, self.y)
    }
}

impl Coord {
    pub const ZERO: Coord = Coord::new(0, 0);
    pub const TOPOLOGY: CoordTopology = CoordTopology;

    pub const fn new(x: i8, y: i8) -> Coord {
        Coord{
            x,
            y,
        }
    }

    pub(super) fn get_hash(&self) -> (u8, u8) {
        let y_r = self.y.rem_euclid(4);
        let y_p = y_r / 2;
        let y_r = y_r % 2;
        let x_r = (self.x + 2 * y_p).rem_euclid(4);
        (x_r as u8, y_r as u8)
    }

    pub fn get_detailed_type(&self) -> DetailedType {
        match self.get_hash() {
            (0,0) => DetailedType::OHex,
            (1,0) => DetailedType::RVoid,
            (2,0) => DetailedType::IPath,
            (3,0) => DetailedType::LVoid,
            (0,1) => DetailedType::AIntersection,
            (1,1) => DetailedType::SPath,
            (2,1) => DetailedType::VIntersection,
            (3,1) => DetailedType::ZPath,
            _ => panic!("Coord has incoherent hash"),
        }
    }

    pub fn get_type(&self) -> Type {
        match self.get_hash() {
            (0,0) => Type::Hex,
            (2,0) | (1,1) | (3,1) => Type::Path,
            (0,1) | (2,1) => Type::Intersection,
            _ => Type::Void,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self {
            Type::Void => "V",
            Type::Hex => "H",
            Type::Path => "P",
            Type::Intersection => "I",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord_new() {
        let coord = Coord::new(5, 7);
        assert_eq!(coord.x, 5);
        assert_eq!(coord.y, 7);
    }

    #[test]
    fn test_coord_zero() {
        assert_eq!(Coord::ZERO.x, 0);
        assert_eq!(Coord::ZERO.y, 0);
    }

    #[test]
    fn test_coord_equality() {
        let a = Coord::new(1, 2);
        let b = Coord::new(1, 2);
        let c = Coord::new(2, 1);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_coord_ordering() {
        let a = Coord::new(1, 1);
        let b = Coord::new(2, 1);
        let c = Coord::new(1, 2);

        // Same y, compare by x
        assert!(a < b);

        // Different y, compare by y first
        assert!(a < c);
        assert!(b < c);
    }

    #[test]
    fn test_coord_get_type_hex() {
        let hex1 = Coord::new(0, 0);
        assert_eq!(hex1.get_type(), Type::Hex);
        let hex2 = Coord::new(4, 0);
        assert_eq!(hex2.get_type(), Type::Hex);
        let hex3 = Coord::new(-2, 2);
        assert_eq!(hex3.get_type(), Type::Hex);
    }

    #[test]
    fn test_coord_get_type_path() {
        let path1 = Coord::new(2, 0);
        assert_eq!(path1.get_type(), Type::Path);

        let path2 = Coord::new(1, 1);
        assert_eq!(path2.get_type(), Type::Path);

        let path3 = Coord::new(3, 1);
        assert_eq!(path3.get_type(), Type::Path);

        let path4 = Coord::new(-3, -1);
        assert_eq!(path4.get_type(), Type::Path);
    }

    #[test]
    fn test_coord_get_type_intersection() {
        let int1 = Coord::new(0, 1);
        assert_eq!(int1.get_type(), Type::Intersection);

        let int2 = Coord::new(2, 1);
        assert_eq!(int2.get_type(), Type::Intersection);

        let int3 = Coord::new(-2, -1);
        assert_eq!(int3.get_type(), Type::Intersection);
    }

    #[test]
    fn test_coord_get_type_void() {
        let void1 = Coord::new(1, 0);
        assert_eq!(void1.get_type(), Type::Void);

        let void2 = Coord::new(3, 0);
        assert_eq!(void2.get_type(), Type::Void);

        let void3 = Coord::new(-1, 0);
        assert_eq!(void3.get_type(), Type::Void);

        let void4 = Coord::new(-3, 2);
        assert_eq!(void4.get_type(), Type::Void);
    }

    #[test]
    fn test_coord_get_detailed_type() {
        assert_eq!(Coord::new(0, 0).get_detailed_type() as u8, DetailedType::OHex as u8);
        assert_eq!(Coord::new(2, 0).get_detailed_type() as u8, DetailedType::IPath as u8);
        assert_eq!(Coord::new(1, 1).get_detailed_type() as u8, DetailedType::SPath as u8);
        assert_eq!(Coord::new(3, 1).get_detailed_type() as u8, DetailedType::ZPath as u8);
        assert_eq!(Coord::new(0, 1).get_detailed_type() as u8, DetailedType::AIntersection as u8);
        assert_eq!(Coord::new(2, 1).get_detailed_type() as u8, DetailedType::VIntersection as u8);
        assert_eq!(Coord::new(1, 0).get_detailed_type() as u8, DetailedType::RVoid as u8);
        assert_eq!(Coord::new(3, 0).get_detailed_type() as u8, DetailedType::LVoid as u8);
    }

    #[test]
    fn test_type_display() {
        assert_eq!(format!("{}", Type::Void), "V");
        assert_eq!(format!("{}", Type::Hex), "H");
        assert_eq!(format!("{}", Type::Path), "P");
        assert_eq!(format!("{}", Type::Intersection), "I");
    }

    #[test]
    fn test_coord_display() {
        let coord = Coord::new(3, 5);
        assert_eq!(format!("{}", coord), "(3,5)");
    }

    #[test]
    fn test_coord_debug() {
        let coord = Coord::new(3, 5);
        assert_eq!(format!("{:?}", coord), "Coord(3,5)");
    }
}
