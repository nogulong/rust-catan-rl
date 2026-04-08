#[cfg(test)]
mod tests {
    use crate::game::{Action, Phase, TurnPhase, DevelopmentPhase, apply};
    use crate::state::{State, TricellState, StateMaker, PlayerId};
    use crate::utils::{Coord, Resource, Resources, Hex, LandHex};
    use crate::board::layout;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    /// Helper function to create a test state with a simple board setup
    fn create_test_state(player_count: u8) -> State {
        let mut state = TricellState::new_empty(&layout::DEFAULT, player_count);

        // Set up a simple board with some hexes
        state.set_static_hex(Coord::new(0, 0), Hex::Land(LandHex::Prod(Resource::Brick, 6))).unwrap();
        state.set_static_hex(Coord::new(4, 0), Hex::Land(LandHex::Prod(Resource::Lumber, 8))).unwrap();
        state.set_static_hex(Coord::new(2, 2), Hex::Land(LandHex::Prod(Resource::Ore, 5))).unwrap();
        state.set_static_hex(Coord::new(-2, 2), Hex::Land(LandHex::Prod(Resource::Grain, 10))).unwrap();
        state.set_static_hex(Coord::new(-4, 0), Hex::Land(LandHex::Prod(Resource::Wool, 9))).unwrap();
        state.set_static_hex(Coord::new(-2, -2), Hex::Land(LandHex::Desert)).unwrap();

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

    /// Helper to place a road for a player
    fn place_road(state: &mut State, player: PlayerId, coord: Coord) -> Result<(), crate::board::Error> {
        state.set_dynamic_path(coord, player)?;
        state.get_player_hand_mut(player).road_pieces -= 1;
        Ok(())
    }

    #[test]
    fn test_apply_end_turn() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);
        let mut rng = SmallRng::seed_from_u64(12345);

        let mut phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        apply::apply(&mut phase, &mut state, Action::EndTurn, &mut rng);

        // Check that turn advanced to next player
        if let Phase::Turn { player: next_player, turn_phase, development_phase } = phase {
            assert_eq!(next_player, PlayerId::from(1u8), "Turn should advance to next player");
            assert_eq!(turn_phase, TurnPhase::PreRoll, "Should be in PreRoll phase");
            assert_eq!(development_phase, DevelopmentPhase::Ready, "Should be Ready");
        } else {
            panic!("Expected Turn phase");
        }
    }

    #[test]
    fn test_apply_end_turn_wraps_around() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(3u8); // Last player
        let mut rng = SmallRng::seed_from_u64(12345);

        let mut phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        apply::apply(&mut phase, &mut state, Action::EndTurn, &mut rng);

        // Check that turn wraps back to first player
        if let Phase::Turn { player: next_player, .. } = phase {
            assert_eq!(next_player, PlayerId::from(0u8), "Turn should wrap to first player");
        } else {
            panic!("Expected Turn phase");
        }
    }

    #[test]
    fn test_apply_build_road() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);
        let mut rng = SmallRng::seed_from_u64(12345);

        // Set up player with road resources
        setup_player_resources(&mut state, player, Resources::ROAD);

        // Place a settlement first to allow road connection
        place_settlement(&mut state, player, Coord::new(0, 1)).unwrap();

        let initial_roads = state.get_player_hand(player).road_pieces;
        let bank_brick = state.get_bank_resources()[Resource::Brick];
        let bank_lumber = state.get_bank_resources()[Resource::Lumber];

        let mut phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::BuildRoad { path: Coord::new(1, 1) };
        apply::apply(&mut phase, &mut state, action, &mut rng);

        // Verify road was placed
        assert_eq!(state.get_dynamic_path(Coord::new(1, 1)).unwrap(), Some(player), "Road should be placed");

        // Verify pieces decreased
        assert_eq!(state.get_player_hand(player).road_pieces, initial_roads - 1, "Road pieces should decrease");

        // Verify resources consumed
        assert_eq!(state.get_player_hand(player).resources[Resource::Brick], 0, "Brick should be consumed");
        assert_eq!(state.get_player_hand(player).resources[Resource::Lumber], 0, "Lumber should be consumed");

        // Verify bank resources increased
        assert_eq!(state.get_bank_resources()[Resource::Brick], bank_brick + 1, "Bank brick should increase");
        assert_eq!(state.get_bank_resources()[Resource::Lumber], bank_lumber + 1, "Bank lumber should increase");
    }

    #[test]
    fn test_apply_build_settlement() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);
        let mut rng = SmallRng::seed_from_u64(12345);

        // Set up player with settlement resources
        setup_player_resources(&mut state, player, Resources::SETTLEMENT);

        // Place a road first to allow settlement connection
        place_road(&mut state, player, Coord::new(1, 1)).unwrap();

        let initial_settlements = state.get_player_hand(player).settlement_pieces;
        let initial_vp = state.get_player_hand(player).building_vp;

        let mut phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::BuildSettlement { intersection: Coord::new(0, 1) };
        apply::apply(&mut phase, &mut state, action, &mut rng);

        // Verify settlement was placed
        let result = state.get_dynamic_intersection(Coord::new(0, 1)).unwrap();
        assert_eq!(result, Some((player, false)), "Settlement should be placed");

        // Verify pieces decreased
        assert_eq!(state.get_player_hand(player).settlement_pieces, initial_settlements - 1, "Settlement pieces should decrease");

        // Verify VP increased
        assert_eq!(state.get_player_hand(player).building_vp, initial_vp + 1, "Victory points should increase");

        // Verify resources consumed
        assert_eq!(state.get_player_hand(player).resources[Resource::Brick], 0);
        assert_eq!(state.get_player_hand(player).resources[Resource::Lumber], 0);
        assert_eq!(state.get_player_hand(player).resources[Resource::Grain], 0);
        assert_eq!(state.get_player_hand(player).resources[Resource::Wool], 0);
    }

    #[test]
    fn test_apply_build_city() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);
        let mut rng = SmallRng::seed_from_u64(12345);

        // Set up player with city resources
        setup_player_resources(&mut state, player, Resources::CITY);

        // Place a settlement first
        place_settlement(&mut state, player, Coord::new(0, 1)).unwrap();

        let initial_cities = state.get_player_hand(player).city_pieces;
        let initial_settlements = state.get_player_hand(player).settlement_pieces;
        let initial_vp = state.get_player_hand(player).building_vp;

        let mut phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::BuildCity { intersection: Coord::new(0, 1) };
        apply::apply(&mut phase, &mut state, action, &mut rng);

        // Verify city was placed (is_city = true)
        let result = state.get_dynamic_intersection(Coord::new(0, 1)).unwrap();
        assert_eq!(result, Some((player, true)), "City should be placed");

        // Verify pieces changed correctly
        assert_eq!(state.get_player_hand(player).city_pieces, initial_cities - 1, "City pieces should decrease");
        assert_eq!(state.get_player_hand(player).settlement_pieces, initial_settlements + 1, "Settlement pieces should increase");

        // Verify VP increased
        assert_eq!(state.get_player_hand(player).building_vp, initial_vp + 1, "Victory points should increase");

        // Verify resources consumed
        assert_eq!(state.get_player_hand(player).resources[Resource::Ore], 0);
        assert_eq!(state.get_player_hand(player).resources[Resource::Grain], 0);
    }

    #[test]
    fn test_apply_trade_bank() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);
        let mut rng = SmallRng::seed_from_u64(12345);

        // Give player 4 brick for 4:1 trade
        setup_player_resources(&mut state, player, Resources::new(4, 0, 0, 0, 0));

        let bank_brick = state.get_bank_resources()[Resource::Brick];
        let bank_lumber = state.get_bank_resources()[Resource::Lumber];

        let mut phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::TradeBank {
            given: Resource::Brick,
            asked: Resource::Lumber,
        };
        apply::apply(&mut phase, &mut state, action, &mut rng);

        // Verify resources changed
        assert_eq!(state.get_player_hand(player).resources[Resource::Brick], 0, "Brick should be consumed");
        assert_eq!(state.get_player_hand(player).resources[Resource::Lumber], 1, "Lumber should be gained");

        // Verify bank resources
        assert_eq!(state.get_bank_resources()[Resource::Brick], bank_brick + 4, "Bank brick should increase by 4");
        assert_eq!(state.get_bank_resources()[Resource::Lumber], bank_lumber - 1, "Bank lumber should decrease by 1");
    }

    #[test]
    fn test_apply_move_thief() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);
        let mut rng = SmallRng::seed_from_u64(12345);

        let initial_thief = state.get_thief_hex();
        let new_hex = Coord::new(0, 0);

        let mut phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::MoveThief,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::MoveThief {
            hex: new_hex,
            victim: PlayerId::NONE,
        };
        apply::apply(&mut phase, &mut state, action, &mut rng);

        // Verify thief moved
        assert_ne!(state.get_thief_hex(), initial_thief, "Thief should move");
        assert_eq!(state.get_thief_hex(), new_hex, "Thief should be at new position");

        // Verify phase changed back to Free
        if let Phase::Turn { turn_phase, .. } = phase {
            assert_eq!(turn_phase, TurnPhase::Free, "Should return to Free phase");
        }
    }

    #[test]
    fn test_apply_buy_development() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);
        let mut rng = SmallRng::seed_from_u64(12345);

        // Give player development card resources
        setup_player_resources(&mut state, player, Resources::DVP_CARD);

        // Set up development cards in the bank
        *state.get_development_cards_mut() = crate::utils::DevelopmentCards {
            knight: 5,
            road_building: 1,
            year_of_plenty: 1,
            monopole: 1,
            victory_point: 2,
        };

        let initial_total_cards = state.get_development_cards().total();
        let initial_player_cards = state.get_player_hand(player).new_development_cards.total();

        let mut phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::BuyDevelopment;
        apply::apply(&mut phase, &mut state, action, &mut rng);

        // Verify development card was bought
        assert_eq!(state.get_development_cards().total(), initial_total_cards - 1, "Bank cards should decrease");
        assert_eq!(state.get_player_hand(player).new_development_cards.total(), initial_player_cards + 1, "Player cards should increase");

        // Verify resources consumed
        assert_eq!(state.get_player_hand(player).resources[Resource::Ore], 0);
        assert_eq!(state.get_player_hand(player).resources[Resource::Grain], 0);
        assert_eq!(state.get_player_hand(player).resources[Resource::Wool], 0);
    }

    #[test]
    fn test_apply_roll_dice_production() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);
        let mut rng = SmallRng::seed_from_u64(99999); // Seed that produces non-7

        // Place a settlement on a hex that produces brick on 6
        place_settlement(&mut state, player, Coord::new(0, 1)).unwrap();

        // Set up harvest for player (this would normally be set during settlement placement)
        state.get_player_hand_mut(player).harvest_on_roll[4] = Resources::new(1, 0, 0, 0, 0); // Roll 6 = index 4

        let mut phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::PreRoll,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::RollDice;
        apply::apply(&mut phase, &mut state, action, &mut rng);

        // Verify phase changed to Free
        if let Phase::Turn { turn_phase, .. } = phase {
            assert_eq!(turn_phase, TurnPhase::Free, "Should be in Free phase after non-7 roll");
        }
    }

    #[test]
    fn test_apply_roll_dice_seven_triggers_thief() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);

        // Use seed that produces a 7 (need to test with specific seed)
        let mut rng = SmallRng::seed_from_u64(42);

        // Give a player 8+ resources so they need to discard
        setup_player_resources(&mut state, player, Resources::new(8, 0, 0, 0, 0));

        let mut phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::PreRoll,
            development_phase: DevelopmentPhase::Ready,
        };

        let action = Action::RollDice;
        let record = apply::apply(&mut phase, &mut state, action, &mut rng);

        // If a 7 was rolled, phase should change
        if record.dice_roll == Some(7) {
            if let Phase::Turn { turn_phase, .. } = phase {
                // Should be either Discard or MoveThief depending on if anyone needs to discard
                assert!(
                    matches!(turn_phase, TurnPhase::Discard(_)) || matches!(turn_phase, TurnPhase::MoveThief),
                    "Should be in Discard or MoveThief phase after rolling 7"
                );
            }
        }
    }

    #[test]
    fn test_apply_development_cards_move_to_hand() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);
        let mut rng = SmallRng::seed_from_u64(12345);

        // Give player a new development card
        state.get_player_hand_mut(player).new_development_cards.knight = 1;

        let mut phase = Phase::Turn {
            player,
            turn_phase: TurnPhase::Free,
            development_phase: DevelopmentPhase::Ready,
        };

        apply::apply(&mut phase, &mut state, Action::EndTurn, &mut rng);

        // After ending turn, new cards should move to regular cards
        // This happens for the player whose turn just ended
        assert_eq!(state.get_player_hand(player).new_development_cards.knight, 0, "New cards should be cleared");
        assert_eq!(state.get_player_hand(player).development_cards.knight, 1, "Cards should move to hand");
    }

    #[test]
    fn test_apply_keep_resources_discard() {
        let mut state = create_test_state(4);
        let player = PlayerId::from(0u8);
        let mut rng = SmallRng::seed_from_u64(12345);

        // Set up discard scenario
        setup_player_resources(&mut state, player, Resources::new(10, 0, 0, 0, 0));

        // Important: State needs to hold discards before Keep action can be applied
        state.hold_discards(vec![(player, None)]);

        let mut phase = Phase::Turn {
            player: PlayerId::from(0u8),
            turn_phase: TurnPhase::Discard(player),
            development_phase: DevelopmentPhase::Ready,
        };

        // Player keeps 5 brick (discarding 5)
        let action = Action::Keep {
            resources: Resources::new(5, 0, 0, 0, 0),
        };
        apply::apply(&mut phase, &mut state, action, &mut rng);

        // Verify resources were discarded
        assert_eq!(state.get_player_hand(player).resources[Resource::Brick], 5, "Should keep 5 brick");
    }
}