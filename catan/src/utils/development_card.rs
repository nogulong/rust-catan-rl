use std::ops::{Index, IndexMut, AddAssign};

/******* DevelopmentCard *******/

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum DevelopmentCard {
    Knight = 0,
    RoadBuilding = 1,
    YearOfPlenty = 2,
    Monopole = 3,
    VictoryPoint = 4,
}

impl DevelopmentCard {
    pub const COUNT: usize = 5;

    pub const ALL: [DevelopmentCard; DevelopmentCard::COUNT] = [
        DevelopmentCard::Knight,
        DevelopmentCard::RoadBuilding,
        DevelopmentCard::YearOfPlenty,
        DevelopmentCard::Monopole,
        DevelopmentCard::VictoryPoint,
    ];

    pub fn to_u8(self) -> u8 {
        self as u8
    }

    pub fn to_usize(self) -> usize {
        self as usize
    }
}

/******* DevelopmentCards *******/

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct DevelopmentCards {
    pub knight: i8,
    pub road_building: i8,
    pub year_of_plenty: i8,
    pub monopole: i8,
    pub victory_point: i8,
}

impl DevelopmentCards {
    pub fn new() -> DevelopmentCards {
        DevelopmentCards {
            knight: 0,
            road_building: 0,
            year_of_plenty: 0,
            monopole: 0,
            victory_point: 0,
        }
    }

    pub fn total(&self) -> u8 {
        (self.knight + self.road_building + self.year_of_plenty + self.monopole + self.victory_point) as u8
    }

    pub fn clear(&mut self) {
        self.knight=0;
        self.road_building=0;
        self.year_of_plenty=0;
        self.monopole=0;
        self.victory_point=0;
    }
}

impl Index<DevelopmentCard> for DevelopmentCards {
    type Output = i8;

    fn index(&self, card: DevelopmentCard) -> &i8 {
        match card {
            DevelopmentCard::Knight => &self.knight,
            DevelopmentCard::RoadBuilding => &self.road_building,
            DevelopmentCard::YearOfPlenty => &self.year_of_plenty,
            DevelopmentCard::Monopole => &self.monopole,
            DevelopmentCard::VictoryPoint => &self.victory_point,
        }
    }
}

impl IndexMut<DevelopmentCard> for DevelopmentCards {
    fn index_mut(&mut self, card: DevelopmentCard) -> &mut i8 {
        match card {
            DevelopmentCard::Knight => &mut self.knight,
            DevelopmentCard::RoadBuilding => &mut self.road_building,
            DevelopmentCard::YearOfPlenty => &mut self.year_of_plenty,
            DevelopmentCard::Monopole => &mut self.monopole,
            DevelopmentCard::VictoryPoint => &mut self.victory_point,
        }
    }
}

impl AddAssign for DevelopmentCards {
    fn add_assign(&mut self, other: DevelopmentCards) {
        self.knight += other.knight;
        self.road_building += other.road_building;
        self.year_of_plenty += other.year_of_plenty;
        self.monopole += other.monopole;
        self.victory_point += other.victory_point;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_development_card_to_u8() {
        assert_eq!(DevelopmentCard::Knight.to_u8(), 0);
        assert_eq!(DevelopmentCard::RoadBuilding.to_u8(), 1);
        assert_eq!(DevelopmentCard::YearOfPlenty.to_u8(), 2);
        assert_eq!(DevelopmentCard::Monopole.to_u8(), 3);
        assert_eq!(DevelopmentCard::VictoryPoint.to_u8(), 4);
    }

    #[test]
    fn test_development_card_to_usize() {
        assert_eq!(DevelopmentCard::Knight.to_usize(), 0);
        assert_eq!(DevelopmentCard::RoadBuilding.to_usize(), 1);
        assert_eq!(DevelopmentCard::YearOfPlenty.to_usize(), 2);
        assert_eq!(DevelopmentCard::Monopole.to_usize(), 3);
        assert_eq!(DevelopmentCard::VictoryPoint.to_usize(), 4);
    }

    #[test]
    fn test_development_cards_new() {
        let cards = DevelopmentCards::new();
        assert_eq!(cards.knight, 0);
        assert_eq!(cards.road_building, 0);
        assert_eq!(cards.year_of_plenty, 0);
        assert_eq!(cards.monopole, 0);
        assert_eq!(cards.victory_point, 0);
    }

    #[test]
    fn test_development_cards_total() {
        let mut cards = DevelopmentCards::new();
        assert_eq!(cards.total(), 0);

        cards.knight = 5;
        cards.road_building = 2;
        cards.year_of_plenty = 2;
        cards.monopole = 2;
        cards.victory_point = 5;
        assert_eq!(cards.total(), 16);
    }

    #[test]
    fn test_development_cards_clear() {
        let mut cards = DevelopmentCards::new();
        cards.knight = 5;
        cards.road_building = 2;
        cards.year_of_plenty = 2;
        cards.monopole = 2;
        cards.victory_point = 5;

        cards.clear();

        assert_eq!(cards.knight, 0);
        assert_eq!(cards.road_building, 0);
        assert_eq!(cards.year_of_plenty, 0);
        assert_eq!(cards.monopole, 0);
        assert_eq!(cards.victory_point, 0);
    }

    #[test]
    fn test_development_cards_index() {
        let mut cards = DevelopmentCards::new();
        cards.knight = 3;
        cards.road_building = 1;
        cards.year_of_plenty = 2;
        cards.monopole = 1;
        cards.victory_point = 4;

        assert_eq!(cards[DevelopmentCard::Knight], 3);
        assert_eq!(cards[DevelopmentCard::RoadBuilding], 1);
        assert_eq!(cards[DevelopmentCard::YearOfPlenty], 2);
        assert_eq!(cards[DevelopmentCard::Monopole], 1);
        assert_eq!(cards[DevelopmentCard::VictoryPoint], 4);
    }

    #[test]
    fn test_development_cards_index_mut() {
        let mut cards = DevelopmentCards::new();
        cards[DevelopmentCard::Knight] = 5;
        cards[DevelopmentCard::VictoryPoint] = 3;

        assert_eq!(cards.knight, 5);
        assert_eq!(cards.victory_point, 3);
    }

    #[test]
    fn test_development_cards_add_assign() {
        let mut cards1 = DevelopmentCards::new();
        cards1.knight = 2;
        cards1.road_building = 1;

        let mut cards2 = DevelopmentCards::new();
        cards2.knight = 3;
        cards2.victory_point = 2;

        cards1 += cards2;

        assert_eq!(cards1.knight, 5);
        assert_eq!(cards1.road_building, 1);
        assert_eq!(cards1.victory_point, 2);
        assert_eq!(cards1.total(), 8);
    }
}