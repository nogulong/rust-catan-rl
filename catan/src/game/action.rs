use crate::utils::{Coord, Resource, Resources, PlayerId};


//typeCatanPlayer= u8;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Action {
    EndTurn,
    RollDice,
    //Discard(Resources),
    MoveThief {
        hex: Coord,
        victim: PlayerId,
    },

    BuildRoad {
        path: Coord
    },
    BuildSettlement {
        intersection: Coord
    },
    BuildCity {
        intersection: Coord
    },

    TradeBank {
        given: Resource,
        asked: Resource
    },
    TradePlayers{
        offer: Resources, 
        want: Resources,
        partner: PlayerId,
    },
    TradePlayersAccept,
    //TradePlayerAlternative(Resources),
    TradePlayersDecline,

    BuyDevelopment,
    DevelopmentKnight,
    DevelopmentRoadBuilding,
    DevelopmentYearOfPlenty {
        resources: Resources,
    },
    DevelopmentMonopole {
        resource: Resource,
    },

    Keep {
        resources: Resources,
    },

    Exit,
    Reset,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum ActionCategory {
    EndTurn = 0,
    RollDice = 1,
    MoveThief = 2,
    BuildRoad = 3,
    BuildSettlement = 4,
    BuildCity = 5,
    TradeBank = 6,
    BuyDevelopment = 7,
    DevelopmentKnight = 8,
    DevelopmentRoadBuilding = 9,
    DevelopmentYearOfPlenty = 10,
    DevelopmentMonopole = 11,
    Keep = 12,
    Exit = 13,
    TradePlayers = 14,
    TradePlayersAccept = 15,
    TradePlayersDecline = 16,
    Reset = 17,
}

impl Action {
    pub fn category(&self) -> ActionCategory {
        match self {
            Action::EndTurn => ActionCategory::EndTurn,
            Action::RollDice => ActionCategory::RollDice,
            Action::MoveThief { hex: _, victim: _ }=> ActionCategory::MoveThief,
            Action::BuildRoad { path: _ } => ActionCategory::BuildRoad,
            Action::BuildSettlement { intersection: _ } => ActionCategory::BuildSettlement,
            Action::BuildCity { intersection: _ } => ActionCategory::BuildCity,
            Action::TradeBank { given: _, asked: _ } => ActionCategory::TradeBank,
            Action::BuyDevelopment => ActionCategory::BuyDevelopment,
            Action::DevelopmentKnight => ActionCategory::DevelopmentKnight,
            Action::DevelopmentRoadBuilding  => ActionCategory::DevelopmentRoadBuilding,
            Action::DevelopmentYearOfPlenty { resources: _ } => ActionCategory::DevelopmentYearOfPlenty,
            Action::DevelopmentMonopole { resource: _ }  => ActionCategory::DevelopmentMonopole,
            Action::Keep { resources: _ } => ActionCategory::Keep,
            Action::Exit => ActionCategory::Exit,
            Action::TradePlayers {offer: _, want: _, partner: _ } => ActionCategory::TradePlayers,
            Action::TradePlayersAccept => ActionCategory::TradePlayersAccept,
            Action::TradePlayersDecline => ActionCategory::TradePlayersDecline,
            Action::Reset => ActionCategory::Reset,
        }
    }
}

impl ActionCategory {
    pub const COUNT: usize = 18;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_category_end_turn() {
        assert_eq!(Action::EndTurn.category() as u8, ActionCategory::EndTurn as u8);
    }

    #[test]
    fn test_action_category_roll_dice() {
        assert_eq!(Action::RollDice.category() as u8, ActionCategory::RollDice as u8);
    }

    #[test]
    fn test_action_category_move_thief() {
        let action = Action::MoveThief {
            hex: Coord::new(0, 0),
            victim: PlayerId::NONE,
        };
        assert_eq!(action.category() as u8, ActionCategory::MoveThief as u8);
    }

    #[test]
    fn test_action_category_build_road() {
        let action = Action::BuildRoad {
            path: Coord::new(2, 0),
        };
        assert_eq!(action.category() as u8, ActionCategory::BuildRoad as u8);
    }

    #[test]
    fn test_action_category_build_settlement() {
        let action = Action::BuildSettlement {
            intersection: Coord::new(0, 1),
        };
        assert_eq!(action.category() as u8, ActionCategory::BuildSettlement as u8);
    }

    #[test]
    fn test_action_category_build_city() {
        let action = Action::BuildCity {
            intersection: Coord::new(0, 1),
        };
        assert_eq!(action.category() as u8, ActionCategory::BuildCity as u8);
    }

    #[test]
    fn test_action_category_trade_bank() {
        let action = Action::TradeBank {
            given: Resource::Brick,
            asked: Resource::Wool,
        };
        assert_eq!(action.category() as u8, ActionCategory::TradeBank as u8);
    }

    #[test]
    fn test_action_category_buy_development() {
        assert_eq!(Action::BuyDevelopment.category() as u8, ActionCategory::BuyDevelopment as u8);
    }

    #[test]
    fn test_action_category_development_cards() {
        assert_eq!(Action::DevelopmentKnight.category() as u8, ActionCategory::DevelopmentKnight as u8);
        assert_eq!(Action::DevelopmentRoadBuilding.category() as u8, ActionCategory::DevelopmentRoadBuilding as u8);

        let year = Action::DevelopmentYearOfPlenty {
            resources: Resources::new(1, 1, 0, 0, 0),
        };
        assert_eq!(year.category() as u8, ActionCategory::DevelopmentYearOfPlenty as u8);

        let mono = Action::DevelopmentMonopole {
            resource: Resource::Grain,
        };
        assert_eq!(mono.category() as u8, ActionCategory::DevelopmentMonopole as u8);
    }

    #[test]
    fn test_action_category_keep() {
        let action = Action::Keep {
            resources: Resources::new(1, 2, 0, 1, 0),
        };
        assert_eq!(action.category() as u8, ActionCategory::Keep as u8);
    }

    #[test]
    fn test_action_category_trade_players() {
        let action = Action::TradePlayers {
            offer: Resources::new(2, 0, 0, 0, 0),
            want: Resources::new(0, 0, 1, 0, 0),
            partner: PlayerId::from(1u8),
        };
        assert_eq!(action.category() as u8, ActionCategory::TradePlayers as u8);
    }

    #[test]
    fn test_action_category_trade_accept_decline() {
        assert_eq!(Action::TradePlayersAccept.category() as u8, ActionCategory::TradePlayersAccept as u8);
        assert_eq!(Action::TradePlayersDecline.category() as u8, ActionCategory::TradePlayersDecline as u8);
    }

    #[test]
    fn test_action_equality() {
        assert_eq!(Action::EndTurn, Action::EndTurn);
        assert_eq!(Action::RollDice, Action::RollDice);
        assert_ne!(Action::EndTurn, Action::RollDice);

        let build1 = Action::BuildRoad { path: Coord::new(2, 0) };
        let build2 = Action::BuildRoad { path: Coord::new(2, 0) };
        let build3 = Action::BuildRoad { path: Coord::new(2, 2) };
        assert_eq!(build1, build2);
        assert_ne!(build1, build3);

        let trade1 = Action::TradeBank { given: Resource::Brick, asked: Resource::Wool };
        let trade2 = Action::TradeBank { given: Resource::Brick, asked: Resource::Wool };
        let trade3 = Action::TradeBank { given: Resource::Grain, asked: Resource::Wool };
        assert_eq!(trade1, trade2);
        assert_ne!(trade1, trade3); 

        let move1 = Action::MoveThief { hex: Coord::new(0, 0), victim: PlayerId::from(1u8) };
        let move2 = Action::MoveThief { hex: Coord::new(0, 0), victim: PlayerId::from(1u8) };
        let move3 = Action::MoveThief { hex: Coord::new(1, 0), victim: PlayerId::from(1u8) };
        let move4 = Action::MoveThief { hex: Coord::new(0, 0), victim: PlayerId::from(2u8) };
        assert_eq!(move1, move2);
        assert_ne!(move1, move3);
        assert_ne!(move1, move4);

        let keep1 = Action::Keep { resources: Resources::new(1, 2, 0, 1, 0) };
        let keep2 = Action::Keep { resources: Resources::new(1, 2, 0, 1, 0) };
        let keep3 = Action::Keep { resources: Resources::new(2, 2, 0, 1, 0) };
        assert_eq!(keep1, keep2);
        assert_ne!(keep1, keep3);

        let tradep1 = Action::TradePlayers {
            offer: Resources::new(2, 0, 0, 0, 0),
            want: Resources::new(0, 0, 1, 0, 0),
            partner: PlayerId::from(1u8),
        };
        let tradep2 = Action::TradePlayers {
            offer: Resources::new(2, 0, 0, 0, 0),
            want: Resources::new(0, 0, 1, 0, 0),
            partner: PlayerId::from(1u8),
        };
        let tradep3 = Action::TradePlayers {
            offer: Resources::new(1, 0, 0, 0, 0),
            want: Resources::new(0, 0, 1, 0, 0),
            partner: PlayerId::from(1u8),
        };
        let tradep4 = Action::TradePlayers {
            offer: Resources::new(2, 0, 0, 0, 0),
            want: Resources::new(0, 1, 1, 0, 0),
            partner: PlayerId::from(1u8),   
        };
        let tradep5 = Action::TradePlayers {
            offer: Resources::new(2, 0, 0, 0, 0),
            want: Resources::new(0, 0, 1, 0, 0),
            partner: PlayerId::from(2u8),   
        };
        assert_eq!(tradep1, tradep2);
        assert_ne!(tradep1, tradep3);
        assert_ne!(tradep1, tradep4);
        assert_ne!(tradep1, tradep5);
    }
}