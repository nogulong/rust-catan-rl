use std::ops::{Index, IndexMut};
use crate::utils::{Resource, Resources, Harbor, DevelopmentCards};

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct AccessibleHarbor {
    harbors: [bool; Harbor::COUNT],
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct PlayerHand {
    pub resources: Resources,
    pub road_pieces: u8,
    pub settlement_pieces: u8,
    pub city_pieces: u8,
    pub building_vp: u8,
    pub knights: u8,
    pub continous_road: u8,
    pub development_cards: DevelopmentCards,
    pub new_development_cards: DevelopmentCards,
    pub harbor: AccessibleHarbor,
    pub harvest_on_roll: [Resources; 11],
    pub blocked_roll: (u8, Resource, i8), // (roll, resource, harvest)
}

impl AccessibleHarbor {
    pub fn new() -> AccessibleHarbor {
        AccessibleHarbor {
            harbors: [false; Harbor::COUNT],
        }
    }

    pub fn rate(&self, resource: Resource) -> u8 {
        let mut required = 4;
        if self[Harbor::Special(resource)] {
            required = 2;
        } else if self[Harbor::Generic] {
            required = 3;
        }
        required
    }

    pub fn add(&mut self, harbor: Harbor) {
        match harbor {
            Harbor::None => return,
            _ => (),
        }
        self[harbor] = true;
    }
}

impl Index<Harbor> for AccessibleHarbor {
    type Output = bool;

    fn index(&self, harbor: Harbor) -> &bool {
         &self.harbors[harbor.to_usize()]
    }
}

impl IndexMut<Harbor> for AccessibleHarbor {
    fn index_mut(&mut self, harbor: Harbor) -> &mut bool {
         &mut self.harbors[harbor.to_usize()]
    }
}

impl Index<usize> for AccessibleHarbor {
    type Output = bool;

    fn index(&self, index: usize) -> &bool {
         &self.harbors[index]
    }
}

impl IndexMut<usize> for AccessibleHarbor {
    fn index_mut(&mut self, index: usize) -> &mut bool {
         &mut self.harbors[index]
    }
}

impl PlayerHand {
    pub fn new() -> PlayerHand {
        PlayerHand {
            resources: Resources::ZERO,
            road_pieces: 15,
            settlement_pieces: 5,
            city_pieces: 4,
            building_vp: 0,
            knights: 0,
            continous_road: 0,
            development_cards: DevelopmentCards::new(),
            new_development_cards: DevelopmentCards::new(),
            harbor: AccessibleHarbor::new(),
            harvest_on_roll: [Resources::ZERO; 11],
            blocked_roll: (0, Resource::Brick, 0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accessible_harbor_new() {
        let harbor = AccessibleHarbor::new();
        for i in 0..Harbor::COUNT {
            assert!(!harbor[i]);
        }
    }

    #[test]
    fn test_accessible_harbor_add() {
        let mut harbor = AccessibleHarbor::new();
        harbor.add(Harbor::Generic);
        assert!(harbor[Harbor::Generic]);

        harbor.add(Harbor::Special(Resource::Brick));
        assert!(harbor[Harbor::Special(Resource::Brick)]);
    }

    #[test]
    fn test_accessible_harbor_rate_default() {
        let harbor = AccessibleHarbor::new();
        assert_eq!(harbor.rate(Resource::Brick), 4);
        assert_eq!(harbor.rate(Resource::Wool), 4);
    }

    #[test]
    fn test_accessible_harbor_rate_generic() {
        let mut harbor = AccessibleHarbor::new();
        harbor.add(Harbor::Generic);
        assert_eq!(harbor.rate(Resource::Brick), 3);
        assert_eq!(harbor.rate(Resource::Wool), 3);
    }

    #[test]
    fn test_accessible_harbor_rate_special() {
        let mut harbor = AccessibleHarbor::new();
        harbor.add(Harbor::Special(Resource::Brick));
        assert_eq!(harbor.rate(Resource::Brick), 2);
        assert_eq!(harbor.rate(Resource::Wool), 4);
    }

    #[test]
    fn test_accessible_harbor_rate_special_overrides_generic() {
        let mut harbor = AccessibleHarbor::new();
        harbor.add(Harbor::Generic);
        harbor.add(Harbor::Special(Resource::Brick));
        assert_eq!(harbor.rate(Resource::Brick), 2);
        assert_eq!(harbor.rate(Resource::Wool), 3);
    }

    #[test]
    fn test_player_hand_new() {
        let hand = PlayerHand::new();
        assert_eq!(hand.resources, Resources::ZERO);
        assert_eq!(hand.road_pieces, 15);
        assert_eq!(hand.settlement_pieces, 5);
        assert_eq!(hand.city_pieces, 4);
        assert_eq!(hand.building_vp, 0);
        assert_eq!(hand.knights, 0);
        assert_eq!(hand.continous_road, 0);
        assert_eq!(hand.development_cards.total(), 0);
        assert_eq!(hand.new_development_cards.total(), 0);
    }

    #[test]
    fn test_player_hand_modify_resources() {
        let mut hand = PlayerHand::new();
        hand.resources = Resources::new(1, 2, 3, 4, 5);
        assert_eq!(hand.resources[Resource::Brick], 1);
        assert_eq!(hand.resources[Resource::Lumber], 2);
        assert_eq!(hand.resources[Resource::Ore], 3);
        assert_eq!(hand.resources[Resource::Grain], 4);
        assert_eq!(hand.resources[Resource::Wool], 5);
    }

    #[test]
    fn test_player_hand_modify_pieces() {
        let mut hand = PlayerHand::new();
        hand.road_pieces = 10;
        hand.settlement_pieces = 3;
        hand.city_pieces = 2;

        assert_eq!(hand.road_pieces, 10);
        assert_eq!(hand.settlement_pieces, 3);
        assert_eq!(hand.city_pieces, 2);
    }

    #[test]
    fn test_player_hand_harvest_on_roll() {
        let mut hand = PlayerHand::new();
        hand.harvest_on_roll[5] = Resources::new(1, 1, 0, 0, 0);
        hand.harvest_on_roll[6] = Resources::new(0, 0, 2, 1, 0);

        assert_eq!(hand.harvest_on_roll[5], Resources::new(1, 1, 0, 0, 0));
        assert_eq!(hand.harvest_on_roll[6], Resources::new(0, 0, 2, 1, 0));
    }
}
