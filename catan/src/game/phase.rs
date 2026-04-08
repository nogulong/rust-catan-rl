use crate::state::PlayerId;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Phase {
    InitialPlacement {
        player: PlayerId,
        placing_second: bool,
        placing_road: bool,
    },
    Turn {
        player: PlayerId,
        turn_phase: TurnPhase,
        development_phase: DevelopmentPhase,
    },
    FinishedGame {
        winner: PlayerId,
    },
}

impl Phase {
    pub const START_GAME: Phase = Phase::InitialPlacement { player: PlayerId::FIRST, placing_second: false, placing_road: false };
    pub const START_TURNS: Phase = Phase::Turn { player: PlayerId::FIRST, turn_phase: TurnPhase::PreRoll, development_phase: DevelopmentPhase::Ready };

    pub fn player(&self) -> PlayerId {
        match self {
            Phase::InitialPlacement { player, placing_second: _, placing_road: _ } => *player,
            Phase::Turn { player: _, turn_phase: TurnPhase::Discard(player), development_phase: _} => *player,
            Phase::Turn {player: _, turn_phase: TurnPhase::TradeSupposed(player), development_phase: _} => *player,
            Phase::Turn { player, turn_phase: _, development_phase: _} => *player,
            Phase::FinishedGame { winner } => *winner,
        }
    }
    pub fn is_turn(&self) -> bool {
        if let Phase::Turn { player: _, turn_phase: _, development_phase: _ } = self {
            true
        } else {
            false
        }
    }

    pub fn is_thief(&self) -> bool {
        if let Phase::Turn { player: _, turn_phase, development_phase } = self {
            *turn_phase == TurnPhase::MoveThief || *development_phase == DevelopmentPhase::KnightActive
        } else {
            false
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TurnPhase {
    PreRoll,
    Discard(PlayerId),
    MoveThief,
    Free,
    TradeSupposed(PlayerId),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DevelopmentPhase {
    Ready,
    KnightActive,
    RoadBuildingActive {
        two_left: bool,
    },
    DevelopmentPlayed,
}

impl TurnPhase {
    pub fn unbound(&self) -> bool {
        match *self {
            TurnPhase::PreRoll | TurnPhase::Free => true,
            _ => false,
        }
    }

    pub fn is_discard(&self) -> bool {
        match *self {
            TurnPhase::Discard(_) => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_start_game() {
        let phase = Phase::START_GAME;
        if let Phase::InitialPlacement { player, placing_second, placing_road } = phase {
            assert_eq!(player, PlayerId::FIRST);
            assert_eq!(placing_second, false);
            assert_eq!(placing_road, false);
        } else {
            panic!("Expected InitialPlacement phase");
        }
    }

    #[test]
    fn test_phase_start_turns() {
        let phase = Phase::START_TURNS;
        if let Phase::Turn { player, turn_phase, development_phase } = phase {
            assert_eq!(player, PlayerId::FIRST);
            assert_eq!(turn_phase, TurnPhase::PreRoll);
            assert_eq!(development_phase, DevelopmentPhase::Ready);
        } else {
            panic!("Expected Turn phase");
        }
    }

    #[test]
    fn test_phase_player_initial() {
        let phase = Phase::InitialPlacement {
            player: PlayerId::from(2u8),
            placing_second: false,
            placing_road: false,
        };
        assert_eq!(phase.player(), PlayerId::from(2u8));
    }

    #[test]
    fn test_phase_player_turn() {
        let phase = Phase::Turn {
            player: PlayerId::from(1u8),
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };
        assert_eq!(phase.player(), PlayerId::from(1u8));
    }

    #[test]
    fn test_phase_player_discard() {
        let phase = Phase::Turn {
            player: PlayerId::from(0u8),
            turn_phase: TurnPhase::Discard(PlayerId::from(3u8)),
            development_phase: DevelopmentPhase::Ready,
        };
        assert_eq!(phase.player(), PlayerId::from(3u8));
    }

    #[test]
    fn test_phase_player_trade_supposed() {
        let phase = Phase::Turn {
            player: PlayerId::from(0u8),
            turn_phase: TurnPhase::TradeSupposed(PlayerId::from(2u8)),
            development_phase: DevelopmentPhase::Ready,
        };
        assert_eq!(phase.player(), PlayerId::from(2u8));
    }

    #[test]
    fn test_phase_is_turn() {
        let turn_phase = Phase::Turn {
            player: PlayerId::FIRST,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };
        assert!(turn_phase.is_turn());

        let initial_phase = Phase::InitialPlacement {
            player: PlayerId::FIRST,
            placing_second: false,
            placing_road: false,
        };
        assert!(!initial_phase.is_turn());
    }

    #[test]
    fn test_phase_is_thief() {
        let thief_phase = Phase::Turn {
            player: PlayerId::FIRST,
            turn_phase: TurnPhase::MoveThief,
            development_phase: DevelopmentPhase::Ready,
        };
        assert!(thief_phase.is_thief());

        let knight_phase = Phase::Turn {
            player: PlayerId::FIRST,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::KnightActive,
        };
        assert!(knight_phase.is_thief());

        let normal_phase = Phase::Turn {
            player: PlayerId::FIRST,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };
        assert!(!normal_phase.is_thief());
    }

    #[test]
    fn test_turn_phase_unbound() {
        assert!(TurnPhase::PreRoll.unbound());
        assert!(TurnPhase::Free.unbound());
        assert!(!TurnPhase::MoveThief.unbound());
        assert!(!TurnPhase::Discard(PlayerId::FIRST).unbound());
        assert!(!TurnPhase::TradeSupposed(PlayerId::FIRST).unbound());
    }

    #[test]
    fn test_turn_phase_is_discard() {
        assert!(TurnPhase::Discard(PlayerId::FIRST).is_discard());
        assert!(!TurnPhase::Free.is_discard());
        assert!(!TurnPhase::PreRoll.is_discard());
        assert!(!TurnPhase::MoveThief.is_discard());
        assert!(!TurnPhase::TradeSupposed(PlayerId::FIRST).is_discard());
    }

    #[test]
    fn test_development_phase_equality() {
        assert_eq!(DevelopmentPhase::Ready, DevelopmentPhase::Ready);
        assert_eq!(DevelopmentPhase::KnightActive, DevelopmentPhase::KnightActive);
        assert_ne!(DevelopmentPhase::Ready, DevelopmentPhase::KnightActive);

        let road1 = DevelopmentPhase::RoadBuildingActive { two_left: true };
        let road2 = DevelopmentPhase::RoadBuildingActive { two_left: true };
        let road3 = DevelopmentPhase::RoadBuildingActive { two_left: false };
        assert_eq!(road1, road2);
        assert_ne!(road1, road3);
    }
}
