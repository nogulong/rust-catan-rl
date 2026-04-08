#[cfg(test)]
mod tests {
    use crate::game::{Action, Phase, TurnPhase, DevelopmentPhase, legal};
    use crate::state::{State, TricellState, StateMaker, PlayerId};
    use crate::utils::{Coord, Resource, Resources, Hex, LandHex, Harbor};
    use crate::board::layout;

    /// Helper function to create a test state with a simple board setup
    fn create_test_state(player_count: u8) -> State {
        let mut state = TricellState::new_empty(&layout::DEFAULT, player_count);

        // Set up a simple board with some hexes
        // Center hex
        state.set_static_hex(Coord::new(0, 0), Hex::Land(LandHex::Prod(Resource::Brick, 6))).unwrap();
        // Adjacent hexes
        state.set_static_hex(Coord::new(4, 0), Hex::Land(LandHex::Prod(Resource::Lumber, 8))).unwrap();
        state.set_static_hex(Coord::new(2, 2), Hex::Land(LandHex::Prod(Resource::Ore, 5))).unwrap();
        state.set_static_hex(Coord::new(-2, 2), Hex::Land(LandHex::Prod(Resource::Grain, 10))).unwrap();
        state.set_static_hex(Coord::new(-4, 0), Hex::Land(LandHex::Prod(Resource::Wool, 9))).unwrap();
        state.set_static_hex(Coord::new(-2, -2), Hex::Land(LandHex::Desert)).unwrap();

        // Initialize thief position on desert
        state.set_thief_hex(Coord::new(-2, -2));

        state
    }

    /// Helper to set up a player with resources
    fn setup_player_resources(state: &mut State, player: PlayerId, resources: Resources) {
        state.get_player_hand_mut(player).resources = resources;
    }

    /// Helper to place a settlement for a player
    fn place_settlement(state: &mut State, player: PlayerId, coord: Coord) -> Result<(), crate::board::Error> {
        state.set_dynamic_intersection(coord, player, false)?;
        state.get_player_hand_mut(player).settlement_pieces -= 1;
        state.get_player_hand_mut(player).building_vp += 1;
        Ok(())
    }

    /// Helper to place a city for a player
    fn place_city(state: &mut State, player: PlayerId, coord: Coord) -> Result<(), crate::board::Error> {
        state.set_dynamic_intersection(coord, player, true)?;
        state.get_player_hand_mut(player).city_pieces -= 1;
        state.get_player_hand_mut(player).settlement_pieces += 1;
        state.get_player_hand_mut(player).building_vp += 1;
        Ok(())
    }

    /// Helper to place a road for a player
    fn place_road(state: &mut State, player: PlayerId, coord: Coord) -> Result<(), crate::board::Error> {
        state.set_dynamic_path(coord, player)?;
        state.get_player_hand_mut(player).road_pieces -= 1;
        Ok(())
    }

    #[test]
    fn test_legal_build_road_no_resources() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Player has no resources
        setup_player_resources(&mut state, player, Resources::ZERO);

        // Place a settlement first
        place_settlement(&mut state, player, Coord::new(0, 1)).unwrap();

        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::BuildRoad { path: Coord::new(1, 1) };
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_err(), "Should not allow building road without resources");
    }

    #[test]
    fn test_legal_build_road_with_resources() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Player has road resources
        setup_player_resources(&mut state, player, Resources::ROAD);

        // Place a settlement and a road first to establish connection
        place_settlement(&mut state, player, Coord::new(0, 1)).unwrap();
        place_road(&mut state, player, Coord::new(1, 1)).unwrap();

        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        // Try to build a second road connected to the first road
        let action = Action::BuildRoad { path: Coord::new(0, 2) };
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_ok(), "Should allow building road with resources and connection");
    }

    #[test]
    fn test_legal_build_settlement_no_resources() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Player has no resources
        setup_player_resources(&mut state, player, Resources::ZERO);

        // Place a road to connect
        place_road(&mut state, player, Coord::new(1, 1)).unwrap();

        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::BuildSettlement { intersection: Coord::new(0, 1) };
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_err(), "Should not allow building settlement without resources");
    }

    #[test]
    fn test_legal_build_settlement_with_resources() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Player has settlement resources
        setup_player_resources(&mut state, player, Resources::SETTLEMENT);

        // Place a road to connect
        place_road(&mut state, player, Coord::new(1, 1)).unwrap();

        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::BuildSettlement { intersection: Coord::new(0, 1) };
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_ok(), "Should allow building settlement with resources and connection");
    }

    #[test]
    fn test_legal_build_settlement_too_close() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Player has settlement resources
        setup_player_resources(&mut state, player, Resources::SETTLEMENT);

        // Place a settlement at one position
        place_settlement(&mut state, player, Coord::new(0, 1)).unwrap();

        // Place roads to connect to adjacent intersection
        place_road(&mut state, player, Coord::new(1, 1)).unwrap();

        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        // Try to place settlement at adjacent intersection (should fail - too close)
        let action = Action::BuildSettlement { intersection: Coord::new(2, 1) };
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_err(), "Should not allow building settlement too close to another");
    }

    #[test]
    fn test_legal_build_city_no_resources() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Player has no resources
        setup_player_resources(&mut state, player, Resources::ZERO);

        // Place a settlement first
        place_settlement(&mut state, player, Coord::new(0, 1)).unwrap();

        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::BuildCity { intersection: Coord::new(0, 1) };
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_err(), "Should not allow building city without resources");
    }

    #[test]
    fn test_legal_build_city_with_resources() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Player has city resources
        setup_player_resources(&mut state, player, Resources::CITY);

        // Place a settlement first
        place_settlement(&mut state, player, Coord::new(0, 1)).unwrap();

        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::BuildCity { intersection: Coord::new(0, 1) };
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_ok(), "Should allow building city with resources on own settlement");
    }

    #[test]
    fn test_legal_build_city_no_settlement() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Player has city resources
        setup_player_resources(&mut state, player, Resources::CITY);

        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        // Try to build city without settlement first
        let action = Action::BuildCity { intersection: Coord::new(0, 1) };
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_err(), "Should not allow building city without settlement");
    }

    #[test]
    fn test_legal_trade_bank_no_resources() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Player has no resources
        setup_player_resources(&mut state, player, Resources::ZERO);

        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::TradeBank {
            given: Resource::Brick,
            asked: Resource::Lumber,
        };
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_err(), "Should not allow trade without enough resources");
    }

    #[test]
    fn test_legal_trade_bank_4_to_1() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Player has 4 brick
        setup_player_resources(&mut state, player, Resources::new(4, 0, 0, 0, 0));

        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::TradeBank {
            given: Resource::Brick,
            asked: Resource::Lumber,
        };
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_ok(), "Should allow 4:1 trade with 4 resources");
    }

    #[test]
    fn test_legal_trade_bank_with_generic_port() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Player has 3 brick and a settlement on a generic port
        setup_player_resources(&mut state, player, Resources::new(3, 0, 0, 0, 0));
        place_settlement(&mut state, player, Coord::new(0, 1)).unwrap(); // Generic port
        state.get_player_hand_mut(player).harbor.add(Harbor::Generic);

        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::TradeBank {
            given: Resource::Brick,
            asked: Resource::Lumber,
        };
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_ok(), "Should allow 3:1 trade with generic port");
    }

    #[test]
    fn test_legal_buy_development_no_resources() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Player has no resources
        setup_player_resources(&mut state, player, Resources::ZERO);

        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::BuyDevelopment;
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_err(), "Should not allow buying development card without resources");
    }

    #[test]
    fn test_legal_buy_development_with_resources() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Player has development card resources
        setup_player_resources(&mut state, player, Resources::DVP_CARD);

        // Make sure there are development cards available
        *state.get_development_cards_mut() = crate::utils::DevelopmentCards {
            knight: 5,
            road_building: 1,
            year_of_plenty: 1,
            monopole: 1,
            victory_point: 2,
        };

        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::BuyDevelopment;
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_ok(), "Should allow buying development card with resources");
    }

    #[test]
    fn test_legal_move_thief() {
        let state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Current thief is on desert at (-2, -2)
        // Try to move to a different hex
        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::MoveThief,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::MoveThief {
            hex: Coord::new(0, 0),
            victim: PlayerId::NONE,
        };
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_ok(), "Should allow moving thief to different hex");
    }

    #[test]
    fn test_legal_move_thief_same_position() {
        let state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Try to move thief to same position
        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::MoveThief,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::MoveThief {
            hex: Coord::new(-2, -2), // Same as current thief position
            victim: PlayerId::NONE,
        };
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_err(), "Should not allow keeping thief in same position");
    }

    #[test]
    fn test_legal_end_turn() {
        let state = create_test_state(4);
        let player = PlayerId::from(0u8);

        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::EndTurn;
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_ok(), "Should allow ending turn in Free phase");
    }

    #[test]
    fn test_legal_roll_dice_wrong_phase() {
        let state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Try to roll dice in Free phase (not PreRoll)
        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::RollDice;
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_err(), "Should not allow rolling dice in Free phase");
    }

    #[test]
    fn test_legal_roll_dice_correct_phase() {
        let state = create_test_state(4);
        let player = PlayerId::from(0u8);

        let phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::PreRoll,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::RollDice;
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_ok(), "Should allow rolling dice in PreRoll phase");
    }

    #[test]
    fn test_legal_initial_placement_settlement() {
        let state = create_test_state(4);
        let player = PlayerId::from(0u8);

        let phase = Phase::InitialPlacement {
            player,
            placing_second: false,
            placing_road: false,
        };

        let action = Action::BuildSettlement { intersection: Coord::new(0, 1) };
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_ok(), "Should allow placing settlement in initial placement");
    }

    #[test]
    fn test_legal_initial_placement_road() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Place a settlement first
        place_settlement(&mut state, player, Coord::new(0, 1)).unwrap();

        let phase = Phase::InitialPlacement {
            player,
            placing_second: false,
            placing_road: true,
        };

        let action = Action::BuildRoad { path: Coord::new(1, 1) };
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_ok(), "Should allow placing road next to settlement in initial placement");
    }
         #[test]
    fn test_legal_trade_accept_partner_insufficient_resources() {
        let mut state = create_test_state(4);
        let turn_player = PlayerId::from(0u8);
        let partner = PlayerId::from(3u8);

        // Turn player has resources to offer
        setup_player_resources(&mut state, turn_player, Resources::new(3, 0, 0, 0, 0));
        
        // Partner does NOT have enough resources
        setup_player_resources(&mut state, partner, Resources::new(0, 0, 0, 0, 0));

        // Set up trade: turn player offers 1 Brick, wants 1 Ore
        let offer = Resources::new(1, 0, 0, 0, 0);
        let wanted = Resources::new(0, 0, 1, 0, 0);
        state.set_trade_info(offer, wanted, turn_player, partner);

        let phase = Phase::Turn {
            player: turn_player,
            turn_phase: TurnPhase::TradeSupposed(partner),
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::TradePlayersAccept;
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_err(), "Should not allow trade when partner lacks resources");
    }

    // #[test]
    // fn test_legal_trade_accept_turn_player_insufficient_resources() {
    //     let mut state = create_test_state(4);
    //     let turn_player = PlayerId::from(0u8);
    //     let partner = PlayerId::from(3u8);

    //     // Turn player does NOT have enough resources to offer
    //     setup_player_resources(&mut state, turn_player, Resources::new(0, 0, 0, 0, 0));
        
    //     // Partner has enough resources
    //     setup_player_resources(&mut state, partner, Resources::new(0, 0, 2, 0, 0));

    //     // Set up trade: turn player offers 1 Brick, wants 1 Ore
    //     let offer = Resources::new(1, 0, 0, 0, 0);
    //     let wanted = Resources::new(0, 0, 1, 0, 0);
    //     state.set_trade_info(offer, wanted, partner);

    //     let phase = Phase::Turn {
    //         player: turn_player,
    //         turn_phase: TurnPhase::TradeSupposed(partner),
    //         development_phase: DevelopmentPhase::Ready,
    //     };

    //     let action = Action::TradePlayersAccept;
    //     let result = legal::legal(&phase, &state, action, true);

    //     assert!(result.is_err(), "Should not allow trade when turn player lacks offered resources");
    // }

    #[test]
    fn test_legal_trade_accept_both_have_resources() {
        let mut state = create_test_state(4);
        let turn_player = PlayerId::from(0u8);
        let partner = PlayerId::from(3u8);

        // Both players have enough resources
        setup_player_resources(&mut state, turn_player, Resources::new(2, 0, 0, 0, 0));
        setup_player_resources(&mut state, partner, Resources::new(0, 0, 2, 0, 0));

        // Set up trade: turn player offers 1 Brick, wants 1 Ore
        let offer = Resources::new(1, 0, 0, 0, 0);
        let wanted = Resources::new(0, 0, 1, 0, 0);
        state.set_trade_info(offer, wanted, turn_player, partner);

        let phase = Phase::Turn {
            player: turn_player,
            turn_phase: TurnPhase::TradeSupposed(partner),
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::TradePlayersAccept;
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_ok(), "Should allow trade when both players have resources");
    }

    #[test]
    fn test_legal_trade_players_propose_with_resources() {
        let mut state = create_test_state(4);
        let turn_player = PlayerId::from(0u8);
        let partner = PlayerId::from(1u8);

        // Turn player has enough resources to offer
        setup_player_resources(&mut state, turn_player, Resources::new(2, 0, 0, 0, 0));

        let phase = Phase::Turn {
            player: turn_player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::TradePlayers {
            offer: Resources::new(1, 0, 0, 0, 0),
            want: Resources::new(0, 0, 1, 0, 0),
            partner,
        };
        
        // Test with skip_trade=false (normal validation)
        let result = legal::legal(&phase, &state, action, true);
        assert!(result.is_ok(), "Should allow trade proposal when player has resources");
    }

    #[test]
    fn test_legal_trade_players_propose_without_resources() {
        let mut state = create_test_state(4);
        let turn_player = PlayerId::from(0u8);
        let partner = PlayerId::from(1u8);

        // Turn player does NOT have enough resources
        setup_player_resources(&mut state, turn_player, Resources::new(0, 0, 0, 0, 0));

        let phase = Phase::Turn {
            player: turn_player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::TradePlayers {
            offer: Resources::new(1, 0, 0, 0, 0),
            want: Resources::new(0, 0, 1, 0, 0),
            partner,
        };
        
        // Test with skip_trade=false (normal validation)
        let result = legal::legal(&phase, &state, action, true);
        assert!(result.is_err(), "Should not allow trade proposal when player lacks resources");
    }

    #[test]
    fn test_legal_trade_players_wrong_phase() {
        let mut state = create_test_state(4);
        let turn_player = PlayerId::from(0u8);
        let partner = PlayerId::from(1u8);

        setup_player_resources(&mut state, turn_player, Resources::new(2, 0, 0, 0, 0));

        // Phase is TradeSupposed, not Free
        let phase = Phase::Turn {
            player: turn_player,
            turn_phase: TurnPhase::TradeSupposed(PlayerId::from(2u8)),
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::TradePlayers {
            offer: Resources::new(1, 0, 0, 0, 0),
            want: Resources::new(0, 0, 1, 0, 0),
            partner,
        };
        
        // Test with skip_trade=false
        let result = legal::legal(&phase, &state, action, true);
        assert!(result.is_err(), "Should not allow trade proposal when not in Free phase");
    }

    #[test]
    fn test_legal_trade_players_invalid_trade() {
        let mut state = create_test_state(4);
        let turn_player = PlayerId::from(0u8);
        let partner = PlayerId::from(1u8);

        setup_player_resources(&mut state, turn_player, Resources::new(2, 0, 0, 0, 0));

        let phase = Phase::Turn {
            player: turn_player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        // Invalid trade: offering and wanting same resource
        let action = Action::TradePlayers {
            offer: Resources::new(1, 0, 0, 0, 0),
            want: Resources::new(1, 0, 0, 0, 0),
            partner,
        };
        
        // Test with skip_trade=false
        let result = legal::legal(&phase, &state, action, true);
        assert!(result.is_err(), "Should not allow invalid trade (same resource)");
    }

    #[test]
    fn test_legal_trade_players_self_trade() {
        let mut state = create_test_state(4);
        let turn_player = PlayerId::from(0u8);

        setup_player_resources(&mut state, turn_player, Resources::new(2, 0, 0, 0, 0));

        let phase = Phase::Turn {
            player: turn_player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        // Try to trade with self
        let action = Action::TradePlayers {
            offer: Resources::new(1, 0, 0, 0, 0),
            want: Resources::new(0, 0, 1, 0, 0),
            partner: turn_player,
        };
        
        // Test with skip_trade=false
        let result = legal::legal(&phase, &state, action, true);
        assert!(result.is_err(), "Should not allow trade with self");
    }

    #[test]
    fn test_legal_trade_accept_wrong_phase() {
        let mut state = create_test_state(4);
        let turn_player = PlayerId::from(0u8);

        setup_player_resources(&mut state, turn_player, Resources::new(2, 0, 0, 0, 0));

        // Phase is Free, not TradeSupposed
        let phase = Phase::Turn {
            player: turn_player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::TradePlayersAccept;
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_err(), "Should not allow accept when not in TradeSupposed phase");
    }

    #[test]
    fn test_legal_trade_decline_valid() {
        let state = create_test_state(4);
        let turn_player = PlayerId::from(0u8);
        let partner = PlayerId::from(1u8);

        let phase = Phase::Turn {
            player: turn_player,
            turn_phase: TurnPhase::TradeSupposed(partner),
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::TradePlayersDecline;
        let result = legal::legal(&phase, &state, action, true);

        assert!(result.is_ok(), "Should allow decline when in TradeSupposed phase");
    }
}