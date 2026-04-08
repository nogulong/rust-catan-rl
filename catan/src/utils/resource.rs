use std::fmt::{self, Display};
use std::ops::{Add, Sub, AddAssign, SubAssign, Index, IndexMut, Neg};
use std::cmp::Ordering;
use std::convert::TryFrom;

/******* Resource *******/

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Resource {
    Brick = 0,
    Lumber = 1,
    Ore = 2,
    Grain = 3,
    Wool = 4,
}

impl Resource {
    pub const COUNT: usize = 5;

    pub const ALL: [Resource; Resource::COUNT] = [
        Resource::Brick,
        Resource::Lumber,
        Resource::Ore,
        Resource::Grain,
        Resource::Wool,
    ];

    pub fn to_u8(self) -> u8 {
        self as u8
    }

    pub fn to_usize(self) -> usize {
        self as usize
    }
}

impl TryFrom<u8> for Resource {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        let value: usize = value as usize;
        if value >= Resource::COUNT {
            Err(())
        } else {
            Ok(Resource::ALL[value])
        }
    }
}

impl Display for Resource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Resource::Brick => write!(f, "B"),
            Resource::Lumber => write!(f, "L"),
            Resource::Ore => write!(f, "O"),
            Resource::Grain => write!(f, "G"),
            Resource::Wool => write!(f, "W"),
        }?;
        Ok(())
    }
}

/******* Resources *******/

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Resources{
    brick: i8,
    lumber: i8,
    ore: i8,
    grain: i8,
    wool: i8,
}

impl Resources {
    pub const ZERO: Resources = Resources::new(0,0,0,0,0);
    pub const ROAD: Resources = Resources::new(1,1,0,0,0);
    pub const SETTLEMENT: Resources = Resources::new(1,1,0,1,1);
    pub const CITY: Resources = Resources::new(0,0,3,2,0);
    pub const DVP_CARD: Resources = Resources::new(0,0,1,1,1);
    pub const STARTING_BANK: Resources = Resources::new(19,19,19,19,19);

    pub const fn new(brick: i8, lumber: i8, ore :i8, grain: i8, wool: i8) -> Self {
        Resources {
            brick,
            lumber,
            ore,
            grain,
            wool
        }
    }

    pub fn new_one(resource: Resource, quantity: i8) -> Self {
        match resource {
            Resource::Brick => Resources::new(quantity, 0, 0, 0, 0),
            Resource::Lumber => Resources::new(0, quantity, 0, 0, 0),
            Resource::Ore => Resources::new(0, 0, quantity, 0, 0),
            Resource::Grain => Resources::new(0, 0, 0, quantity, 0),
            Resource::Wool => Resources::new(0, 0, 0, 0, quantity),
        }
    }

    fn cmp(&self, other: &Resources) -> [Ordering; Resource::COUNT] {[
        self.brick.cmp(&other.brick),
        self.lumber.cmp(&other.lumber),
        self.ore.cmp(&other.ore),
        self.grain.cmp(&other.grain),
        self.wool.cmp(&other.wool),
    ]}

    pub fn total(&self) -> i8 {
        self.brick + self.lumber + self.ore + self.grain + self.wool
    }

    pub fn valid_trade(&self) -> bool {
        !(self >= &Resources::ZERO) && !(self <= &Resources::ZERO)
    }
}

impl Add for Resources {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Resources::new(
            self.brick + other.brick,
            self.lumber + other.lumber,
            self.ore + other.ore,
            self.grain + other.grain,
            self.wool + other.wool,
        )
    }
}

impl Sub for Resources {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Resources::new(
            self.brick - other.brick,
            self.lumber - other.lumber,
            self.ore - other.ore,
            self.grain - other.grain,
            self.wool - other.wool,
        )
    }
}

impl AddAssign for Resources {
    fn add_assign(&mut self, other: Self) {
        self.brick += other.brick;
        self.lumber += other.lumber;
        self.ore += other.ore;
        self.grain += other.grain;
        self.wool += other.wool;
    }
}

impl SubAssign for Resources {
    fn sub_assign(&mut self, other: Self) {
        self.brick -= other.brick;
        self.lumber -= other.lumber;
        self.ore -= other.ore;
        self.grain -= other.grain;
        self.wool -= other.wool;
    }
}

impl Index<Resource> for Resources {
    type Output = i8;

    fn index(&self, resource: Resource) -> &i8 {
        match resource {
            Resource::Brick => &self.brick,
            Resource::Lumber => &self.lumber,
            Resource::Ore => &self.ore,
            Resource::Grain => &self.grain,
            Resource::Wool => &self.wool,
        }
    }
}

impl IndexMut<Resource> for Resources {
    fn index_mut(&mut self, resource: Resource) -> &mut i8 {
        match resource {
            Resource::Brick => &mut self.brick,
            Resource::Lumber => &mut self.lumber,
            Resource::Ore => &mut self.ore,
            Resource::Grain => &mut self.grain,
            Resource::Wool => &mut self.wool,
        }
    }
}

impl Index<usize> for Resources {
    type Output = i8;

    fn index(&self, resource: usize) -> &i8 {
        match resource {
            0 => &self.brick,
            1 => &self.lumber,
            2 => &self.ore,
            3 => &self.grain,
            4 => &self.wool,
            _ => panic!(),
        }
    }
}

impl IndexMut<usize> for Resources {
    fn index_mut(&mut self, resource: usize) -> &mut i8 {
        match resource {
            0 => &mut self.brick,
            1 => &mut self.lumber,
            2 => &mut self.ore,
            3 => &mut self.grain,
            4 => &mut self.wool,
            _ => panic!(),
        }
    }
}

impl PartialOrd for Resources {
    fn partial_cmp(&self, other: &Resources) -> Option<Ordering> {
        let orderings = self.cmp(other);
        let mut current = Ordering::Equal;
        for ordering in orderings.iter() {
            match ordering {
                Ordering::Equal => (),
                Ordering::Greater => {
                    if current == Ordering::Less {
                        return None;
                    }
                    current = Ordering::Greater;
                }
                Ordering::Less => {
                    if current == Ordering::Greater {
                        return None;
                    }
                    current = Ordering::Less;

                }
            }
        };
        Some(current)
    }
}

impl Neg for Resources {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Resources::new(
            -self.brick,
            -self.lumber,
            -self.ore,
            -self.grain,
            -self.wool,
        )
    }
}

// 参照に対するNegも実装（便利のため）
impl Neg for &Resources {
    type Output = Resources;

    fn neg(self) -> Self::Output {
        Resources::new(
            -self.brick,
            -self.lumber,
            -self.ore,
            -self.grain,
            -self.wool,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_to_u8() {
        assert_eq!(Resource::Brick.to_u8(), 0);
        assert_eq!(Resource::Lumber.to_u8(), 1);
        assert_eq!(Resource::Ore.to_u8(), 2);
        assert_eq!(Resource::Grain.to_u8(), 3);
        assert_eq!(Resource::Wool.to_u8(), 4);
    }

    #[test]
    fn test_resource_try_from() {
        assert_eq!(Resource::try_from(0), Ok(Resource::Brick));
        assert_eq!(Resource::try_from(1), Ok(Resource::Lumber));
        assert_eq!(Resource::try_from(2), Ok(Resource::Ore));
        assert_eq!(Resource::try_from(3), Ok(Resource::Grain));
        assert_eq!(Resource::try_from(4), Ok(Resource::Wool));
        assert!(Resource::try_from(5).is_err());
        assert!(Resource::try_from(10).is_err());
    }

    #[test]
    fn test_resources_constants() {
        assert_eq!(Resources::ZERO, Resources::new(0, 0, 0, 0, 0));
        assert_eq!(Resources::ROAD, Resources::new(1, 1, 0, 0, 0));
        assert_eq!(Resources::SETTLEMENT, Resources::new(1, 1, 0, 1, 1));
        assert_eq!(Resources::CITY, Resources::new(0, 0, 3, 2, 0));
        assert_eq!(Resources::DVP_CARD, Resources::new(0, 0, 1, 1, 1));
    }

    #[test]
    fn test_resources_new_one() {
        let brick_res = Resources::new_one(Resource::Brick, 3);
        assert_eq!(brick_res, Resources::new(3, 0, 0, 0, 0));

        let lumber_res = Resources::new_one(Resource::Lumber, 5);
        assert_eq!(lumber_res, Resources::new(0, 5, 0, 0, 0));
    }

    #[test]
    fn test_resources_total() {
        assert_eq!(Resources::ZERO.total(), 0);
        assert_eq!(Resources::ROAD.total(), 2);
        assert_eq!(Resources::SETTLEMENT.total(), 4);
        assert_eq!(Resources::CITY.total(), 5);
        assert_eq!(Resources::new(1, 2, 3, 4, 5).total(), 15);
    }

    #[test]
    fn test_resources_add() {
        let a = Resources::new(1, 2, 3, 4, 5);
        let b = Resources::new(5, 4, 3, 2, 1);
        let result = a + b;
        assert_eq!(result, Resources::new(6, 6, 6, 6, 6));
    }

    #[test]
    fn test_resources_sub() {
        let a = Resources::new(5, 4, 3, 2, 1);
        let b = Resources::new(1, 2, 3, 4, 5);
        let result = a - b;
        assert_eq!(result, Resources::new(4, 2, 0, -2, -4));
    }

    #[test]
    fn test_resources_add_assign() {
        let mut res = Resources::new(1, 1, 1, 1, 1);
        res += Resources::new(2, 3, 4, 5, 6);
        assert_eq!(res, Resources::new(3, 4, 5, 6, 7));
    }

    #[test]
    fn test_resources_sub_assign() {
        let mut res = Resources::new(5, 5, 5, 5, 5);
        res -= Resources::new(1, 2, 3, 4, 5);
        assert_eq!(res, Resources::new(4, 3, 2, 1, 0));
    }

    #[test]
    fn test_resources_index_resource() {
        let res = Resources::new(1, 2, 3, 4, 5);
        assert_eq!(res[Resource::Brick], 1);
        assert_eq!(res[Resource::Lumber], 2);
        assert_eq!(res[Resource::Ore], 3);
        assert_eq!(res[Resource::Grain], 4);
        assert_eq!(res[Resource::Wool], 5);
    }

    #[test]
    fn test_resources_index_usize() {
        let res = Resources::new(1, 2, 3, 4, 5);
        assert_eq!(res[0], 1);
        assert_eq!(res[1], 2);
        assert_eq!(res[2], 3);
        assert_eq!(res[3], 4);
        assert_eq!(res[4], 5);
    }

    #[test]
    fn test_resources_index_mut() {
        let mut res = Resources::ZERO;
        res[Resource::Brick] = 5;
        res[Resource::Lumber] = 3;
        assert_eq!(res, Resources::new(5, 3, 0, 0, 0));
    }

    #[test]
    fn test_resources_partial_ord() {
        let a = Resources::new(1, 2, 3, 4, 5);
        let b = Resources::new(1, 2, 3, 4, 5);
        let c = Resources::new(2, 3, 4, 5, 6);
        let d = Resources::new(0, 1, 2, 3, 4);
        let e = Resources::new(2, 1, 3, 4, 5); // Some greater, some less

        // Equal case
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Equal));
        assert!(a == b);

        // All less
        assert_eq!(a.partial_cmp(&c), Some(Ordering::Less));
        assert!(a < c);

        // All greater
        assert_eq!(a.partial_cmp(&d), Some(Ordering::Greater));
        assert!(a > d);

        // Mixed - should return None
        assert_eq!(a.partial_cmp(&e), None);
        assert!(!(a < e));
        assert!(!(a > e));
        assert!(!(a == e));
    }

    #[test]
    fn test_resources_neg() {
        let res = Resources::new(1, 2, 3, -4, -5);
        let neg_res = -res;
        assert_eq!(neg_res, Resources::new(-1, -2, -3, 4, 5));
    }

    #[test]
    fn test_resources_neg_ref() {
        let res = Resources::new(1, 2, 3, -4, -5);
        let neg_res = -&res;
        assert_eq!(neg_res, Resources::new(-1, -2, -3, 4, 5));
    }

    #[test]
    fn test_resources_valid_trade() {
        // Valid trade: some positive, some negative
        let valid = Resources::new(1, 2, -1, -1, 0);
        assert!(valid.valid_trade());

        // Invalid trade: all positive
        let all_positive = Resources::new(1, 2, 3, 4, 5);
        assert!(!all_positive.valid_trade());

        // Invalid trade: all negative
        let all_negative = Resources::new(-1, -2, -3, -4, -5);
        assert!(!all_negative.valid_trade());

        // Invalid trade: all zero
        assert!(!Resources::ZERO.valid_trade());
    }
}