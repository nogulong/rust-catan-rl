mod development_card;
mod resource;

pub use development_card::{DevelopmentCard, DevelopmentCards};
pub use resource::{Resource, Resources};
pub use crate::board::{Coord, CoordType};
pub use crate::state::PlayerId;

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Empty;

impl Empty {
    pub const INSTANCE: Empty = Empty {};
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Hex {
    Water,
    Land(LandHex),
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum LandHex {
    Prod(Resource, u8),
    Desert,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Harbor {
    None,
    Generic,
    Special(Resource),
}

impl Hex {
    pub fn get_num(&self) -> Option<u8> {
        match self {
            Hex::Land(land) => match land {
                LandHex::Desert => None,
                LandHex::Prod(_, val) => Some(*val),
            },
            _ => None,
        }
    }
}

impl Harbor {
    pub const COUNT: usize = 6;

    pub fn to_usize(self) -> usize {
        match self {
            Harbor::None => 6,
            Harbor::Generic => 5,
            Harbor::Special(res) => {
                res as usize
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_instance() {
        let e1 = Empty::INSTANCE;
        let e2 = Empty::INSTANCE;
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_hex_water() {
        let hex = Hex::Water;
        assert_eq!(hex.get_num(), None);
    }

    #[test]
    fn test_hex_desert() {
        let hex = Hex::Land(LandHex::Desert);
        assert_eq!(hex.get_num(), None);
    }

    #[test]
    fn test_hex_production() {
        let hex = Hex::Land(LandHex::Prod(Resource::Brick, 6));
        assert_eq!(hex.get_num(), Some(6));

        let hex2 = Hex::Land(LandHex::Prod(Resource::Grain, 8));
        assert_eq!(hex2.get_num(), Some(8));
    }

    #[test]
    fn test_hex_equality() {
        let h1 = Hex::Land(LandHex::Prod(Resource::Wool, 5));
        let h2 = Hex::Land(LandHex::Prod(Resource::Wool, 5));
        let h3 = Hex::Land(LandHex::Prod(Resource::Wool, 6));

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_land_hex_equality() {
        let l1 = LandHex::Desert;
        let l2 = LandHex::Desert;
        let l3 = LandHex::Prod(Resource::Brick, 4);

        assert_eq!(l1, l2);
        assert_ne!(l1, l3);
    }

    #[test]
    fn test_harbor_to_usize() {
        assert_eq!(Harbor::None.to_usize(), 6);
        assert_eq!(Harbor::Generic.to_usize(), 5);
        assert_eq!(Harbor::Special(Resource::Brick).to_usize(), 0);
        assert_eq!(Harbor::Special(Resource::Lumber).to_usize(), 1);
        assert_eq!(Harbor::Special(Resource::Ore).to_usize(), 2);
        assert_eq!(Harbor::Special(Resource::Grain).to_usize(), 3);
        assert_eq!(Harbor::Special(Resource::Wool).to_usize(), 4);
    }

    #[test]
    fn test_harbor_count() {
        assert_eq!(Harbor::COUNT, 6);
    }

    #[test]
    fn test_harbor_equality() {
        let h1 = Harbor::Generic;
        let h2 = Harbor::Generic;
        let h3 = Harbor::None;
        let h4 = Harbor::Special(Resource::Brick);
        let h5 = Harbor::Special(Resource::Brick);

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert_eq!(h4, h5);
        assert_ne!(h1, h4);
    }
}
