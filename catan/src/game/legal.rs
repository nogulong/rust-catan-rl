use crate::utils::{Coord, CoordType, DevelopmentCard, Resource, Resources};
use crate::state::{State, PlayerId};
use crate::game::{Phase, TurnPhase, DevelopmentPhase, Action, Error};
use crate::board::utils::topology::Topology;
use crate::board::Error as BoardError;

/// Is the intersection free for a settlement
///
/// Returns true if there is no settlement at the coord or around it
pub fn available_settlement_position(coord: Coord, state: &State) -> Result<bool, BoardError> {
    let neighbours = state.intersection_intersection_neighbours(coord)?;
    for neighbour_intersection in neighbours {
        if state.get_dynamic_intersection(neighbour_intersection)?.is_some() {
            return Ok(false);
        }
    }
    return Ok(state.get_dynamic_intersection(coord)?.is_none());
}

/// Is this position allowed for a inital placement road
///
/// Returns true if the path or intersection coord is next to a road owned by the player
pub fn allowed_initial_road_placement(coord: Coord, player: PlayerId, state: &State) -> Result<bool, BoardError> {
    let neighbours = state.path_intersection_neighbours(coord)?;
    let mut neighbour_settlement = None;
    for neighbour in neighbours {
        if let Some((p, _)) = state.get_dynamic_intersection(neighbour)? {
            if player == p {
                neighbour_settlement = Some(neighbour);
            }
        }
    }
    if let Some(neighbour_settlement) = neighbour_settlement {
        let connected = connected_position(neighbour_settlement, player, state)?;
        // If the settlement is already connected it means we are putting the player is placing the road next to the wrong selltement
        Ok(!connected)
    } else {
        Ok(false)
    }
}

/// Does this victim have a settlement or city around the hex
///
/// Useful to check if the player can steal from the victim
pub fn can_steal_victim(player: PlayerId, target_hex: Coord, victim: PlayerId, state: &State) -> Result<(), Error> {
    let mut potential_victims = vec![false; state.player_count() as usize];
    let mut no_victim_available = true;
    for intersection in state.hex_intersection_neighbours(target_hex)?.iter() {
        if let Some((p, _)) = state.get_dynamic_intersection(*intersection)? {
            if p != player {
                potential_victims[p.to_usize()] = true;
                no_victim_available = false;
            }
        }
    }
    if victim == PlayerId::NONE || victim == player {// Steal from no one
        if no_victim_available {
            return Ok(());
        } else {
            return Err(Error::MustPickVictim);
        }
    } else if victim.to_u8() >= state.player_count() { // index out of range
        return Err(Error::WrongVictim { victim });
    } else if potential_victims[victim.to_usize()] { // The victim has a settlement or city around the hex
        return Ok(());
    } else { // The victim has no settlement or city around the hex
        return Err(Error::WrongVictim { victim });
    }
}

/// Can put road
///
/// Can the player put a road at the given path
/// Checks number of road pieces left, if the position is connected and if the position is empty
/// But NOT the player's resources
///  + should check if the mid intersection is not occupied by another player's settlement/city
pub fn can_put_road(player: PlayerId, path: Coord, state: &State) -> Result<(), Error> {
    // Does the player have a road piece left?
    if state.get_player_hand(player).road_pieces == 0 {
        Err(Error::NoMorePiece { piece: 0 })
    // Is the path are next to a road owned by the player?
    } else if !connected_position(path, player, state)? {
        Err(Error::NotConnected { coord: path })
    // Is the path blocked by another player's settlement/city at the mid intersection?
    } else if is_blocked_path(path, player, state) {
        Err(Error::NotConnected { coord: path })
    // Is the position empty?
    } else if state.get_dynamic_path(path)?.is_some() {
        Err(Error::AlreadyOccupied { coord: path })
    } else {
        Ok(())
    }
}

/// Is the path or intersection connected to a piece owned by the player
///
/// Returns true if the path or intersection coord is next to a road owned by the player
pub fn connected_position(coord: Coord, player: PlayerId, state: &State) -> Result<bool, BoardError> {
    let neighbours = match coord.get_type() {
        CoordType::Path => state.path_path_neighbours(coord)?,
        CoordType::Intersection => state.intersection_path_neighbours(coord)?,
        t => return Err(BoardError::MultiWrongCoordType { expected:[false, false, true, true], received:t }),
    };
    for neighbour in neighbours {
        if let Some(p) = state.get_dynamic_path(neighbour)? {
            if player == p {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Is the path blocked by another player's settlement/city at the mid intersection
pub fn is_blocked_path(path: Coord, player: PlayerId, state: &State) -> bool {
    let neighbour_intersections = match state.path_intersection_neighbours(path) {
        Ok(neighbours) => neighbours,
        Err(_) => return false,
    };
    // 接続元の交差点があるかチェック
    for intersection in neighbour_intersections {
        // 誰の頂点でもない -> その頂点に接続する道があればいい
        // 他のプレイヤーの頂点 -> ブロックされている
        if let Some((p, _)) = state.get_dynamic_intersection(intersection).expect("Getting intersection failed") {
            if p != player {
                continue; // 他のプレイヤーの頂点ならブロックされている
            } else {
                return false; // 自分の頂点ならブロックされていない
            }
        } else { // 誰の頂点でもない
            let neighbour_paths = state.intersection_path_neighbours(intersection).expect("Getting intersection neighbours failed");
            for neighbour_path in neighbour_paths {
                if let Some(p) = state.get_dynamic_path(neighbour_path).expect("Getting path failed") {
                    if p == player {
                        return false; // 自分の道に接続しているならブロックされていない
                    }
                }
            }
        }
    }
    true // 道に接続する頂点がすべてブロックされている, もしくは接続する道がない
}

/// Is the action legal in this context
///
/// Returns either an ok if the action can be played in the current phase and state,
/// or an Error describing why the action can't be played
pub fn legal(phase: &Phase, state: &State, action: Action, trade_allowed: bool) -> Result<(), Error> {
    match phase {
        //
        // # Initial Placement Phase
        //
        Phase::InitialPlacement { player, placing_second: _, placing_road } => {
            //
            // ## Building a Settlement
            //
            if !*placing_road {
                if let Action::BuildSettlement { intersection } = action {
                    // If the position is available
                    if available_settlement_position(intersection, state)? {
                        Ok(())
                    } else {
                        Err(Error::IllegalAction(action))
                    }
                } else {
                    Err(Error::IncoherentAction(action))
                }
            //
            // ## Building a Road
            //
            } else {
                if let Action::BuildRoad { path } = action {
                    if allowed_initial_road_placement(path, *player, state)? {
                        // We don't have to check it is unoccupied because the Settlement could not have been placed next to another existing road
                        Ok(())
                    } else {
                        Err(Error::IllegalAction(action))
                    }
                } else {
                    Err(Error::IncoherentAction(action))
                }
            }
        }
        //
        // # Regular Turn Phase
        //
        Phase::Turn { player, turn_phase, development_phase } => match action {
            //
            // ## Ending Turn
            //
            Action::EndTurn => {
                if *turn_phase == TurnPhase::Free && *development_phase != DevelopmentPhase::KnightActive{
                    Ok(())
                } else {
                    Err(Error::IncoherentAction(action))
                }
            }
            //
            // ## Rolling Dice
            //
            Action::RollDice => {
                if *turn_phase == TurnPhase::PreRoll && (*development_phase == DevelopmentPhase::Ready || *development_phase == DevelopmentPhase::DevelopmentPlayed){
                    Ok(())
                } else {
                    Err(Error::IncoherentAction(action))
                }
            }
            //
            // ## Discarding
            //
            Action::Keep { resources } => {
                if let TurnPhase::Discard(player) = *turn_phase {
                    let current = state.get_player_hand(player).resources;
                    if current >= resources && resources.total() <= (current.total()+1) / 2 {
                        Ok(())
                    } else {
                        Err(Error::NotEnoughResources {
                            required: resources,
                            have: current,
                        })
                    }
                } else {
                    Err(Error::IncoherentAction(action))
                }
            }
            //
            // ## Moving Thief
            //
            Action::MoveThief { hex, victim } => {
                if *turn_phase == TurnPhase::MoveThief || (!turn_phase.is_discard() && *development_phase == DevelopmentPhase::KnightActive) {
                    if hex == state.get_thief_hex() {
                        Err(Error::ThiefNotMoved { hex })
                    } else {
                        can_steal_victim(*player, hex, victim, state)
                    }
                } else {
                    Err(Error::IncoherentAction(action))
                }
            }
            //
            // ## Building Road
            //
            Action::BuildRoad { path } => {
                let road_building = if let DevelopmentPhase::RoadBuildingActive { two_left: _ } = *development_phase {
                    *turn_phase == TurnPhase::PreRoll || *turn_phase == TurnPhase::Free
                } else {
                    false
                };
                if (*turn_phase != TurnPhase::Free && !road_building) || (*development_phase == DevelopmentPhase::KnightActive) {
                    return Err(Error::IncoherentAction(action));
                }
                can_put_road(*player, path, state)?;
                if road_building || state.get_player_hand(*player).resources >= Resources::ROAD {
                    Ok(())
                } else {
                    Err(Error::NotEnoughResources {
                        required: Resources::ROAD,
                        have: state.get_player_hand(*player).resources
                    })
                }
            }
            //
            // ## Building Settlement
            //
            Action::BuildSettlement { intersection } => {
                if *turn_phase != TurnPhase::Free || (*development_phase != DevelopmentPhase::Ready && *development_phase != DevelopmentPhase::DevelopmentPlayed) {
                    return Err(Error::IncoherentAction(action));
                }
                // If: we are next to a road...
                if connected_position(intersection, *player, state)?
                    // ...the position is available (no settlement on it or next to it)...
                    && available_settlement_position(intersection, state)?
                    // ...the player has a settlement piece left...
                    && state.get_player_hand(*player).settlement_pieces >= 1
                    // ...and the player has enough resources for the settlement
                    && state.get_player_hand(*player).resources >= Resources::SETTLEMENT {
                    Ok(())
                } else {
                    Err(Error::IllegalAction(action))
                }
            }
            //
            // ## Building City
            //
            Action::BuildCity { intersection } => {
                if *turn_phase != TurnPhase::Free || (*development_phase != DevelopmentPhase::Ready && *development_phase != DevelopmentPhase::DevelopmentPlayed) {
                    return Err(Error::IncoherentAction(action));
                }
                // If: we already own a settlement at the position
                if Some((*player, false)) == state.get_dynamic_intersection(intersection)?
                    // ...and the player has a city piece left...
                    && state.get_player_hand(*player).city_pieces >= 1
                    // ...and the player has enough resources for the city
                    && state.get_player_hand(*player).resources >= Resources::CITY {
                    Ok(())
                } else {
                    Err(Error::IllegalAction(action))
                }
            }
            //
            // ## Trade Bank
            //
            Action::TradeBank { given, asked } => {
                if *turn_phase != TurnPhase::Free || (*development_phase != DevelopmentPhase::Ready && *development_phase != DevelopmentPhase::DevelopmentPlayed) {
                    return Err(Error::IncoherentAction(action));
                }
                if given == asked {
                    return Err(Error::IllegalTradeSameResources(given));
                }
                let hand = state.get_player_hand(*player);
                let rate = hand.harbor.rate(given);
                if hand.resources[given] < rate as i8 {
                    Err(Error::NotEnoughResources { required: Resources::new_one(given, rate as i8), have: hand.resources })
                } else if state.get_bank_resources()[asked] <= 0 {
                    Err(Error::NoMoreResourceInBank(asked))
                } else {
                    Ok(())
                }
            }
            //
            // Trade with players
            //
            Action::TradePlayers {offer, want, partner} => {
                if !trade_allowed {
                    return Err(Error::IllegalAction(action));
                }
                if *turn_phase != TurnPhase::Free || (*development_phase != DevelopmentPhase::Ready && *development_phase != DevelopmentPhase::DevelopmentPlayed) {
                    return Err(Error::IncoherentAction(action));
                }
                if partner == *player || partner.to_u8() >= state.player_count(){
                    return Err(Error::IllegalAction(action));
                }
                let diff = want - offer;
                if !diff.valid_trade() {
                    return Err(Error::IllegalAction(action));
                }
                let current = state.get_player_hand(*player).resources;
                if current >= offer {
                    Ok(())
                } else {
                    Err(Error::NotEnoughResources { required: offer, have: current })
                }
            }
            //
            // Trade Accept
            //
            Action::TradePlayersAccept => {
                if let TurnPhase::TradeSupposed(player) = *turn_phase {
                    let current = state.get_player_hand(player).resources;
                    let wanted = state.get_trade_wanted();
                    if current >= wanted {
                        Ok(())
                    } else {
                        Err(Error::NotEnoughResources { required: wanted, have: current })
                    }
                } else {
                    Err(Error::IncoherentAction(action))
                }
            }
            //
            //Trade Decline
            //
            Action::TradePlayersDecline => {
                if let TurnPhase::TradeSupposed(_) = *turn_phase {
                    Ok(())
                } else {
                    Err(Error::IncoherentAction(action))
                }
            }
            //
            // ## Buy Development Card
            //
            Action::BuyDevelopment => {
                if *turn_phase != TurnPhase::Free || (*development_phase != DevelopmentPhase::Ready && *development_phase != DevelopmentPhase::DevelopmentPlayed) {
                    return Err(Error::IncoherentAction(action));
                }
                if state.get_development_cards().total() >= 1
                    && state.get_player_hand(*player).resources >= Resources::DVP_CARD {
                        Ok(())
                } else {
                    Err(Error::IllegalAction(action))
                }
            }
            //
            // ## Use Knight Development Card
            //
            Action::DevelopmentKnight => {
                if !turn_phase.unbound() {
                    Err(Error::IncoherentAction(action))
                } else if *development_phase != DevelopmentPhase::Ready {
                    Err(Error::DevelopmentCardAlreadyPlayed)
                } else if state.get_player_hand(*player).development_cards.knight == 0 {
                    Err(Error::NoCard { card_type: DevelopmentCard::Knight })
                } else {
                    Ok(())
                }
            }
            //
            // ## Use Road Building Development Card
            //
            Action::DevelopmentRoadBuilding => {
                if !turn_phase.unbound() {
                    Err(Error::IncoherentAction(action))
                } else if *turn_phase == TurnPhase::PreRoll{
                    Err(Error::IncoherentAction(action))
                } else if *development_phase != DevelopmentPhase::Ready {
                    Err(Error::DevelopmentCardAlreadyPlayed)
                } else if state.get_player_hand(*player).development_cards.road_building == 0 {
                    Err(Error::NoCard { card_type: DevelopmentCard::RoadBuilding })
                } else {
                    Ok(())
                }
            }
            //
            // ## Use Year of Plenty Development Card
            //
            Action::DevelopmentYearOfPlenty { resources } => {
                if !turn_phase.unbound() {
                    Err(Error::IncoherentAction(action))
                } else if *development_phase != DevelopmentPhase::Ready {
                    Err(Error::DevelopmentCardAlreadyPlayed)
                } else if state.get_player_hand(*player).development_cards.year_of_plenty == 0 {
                    Err(Error::NoCard { card_type: DevelopmentCard::YearOfPlenty })
                } else if resources.total() != 2 {
                    Err(Error::IllegalAction(action))
                } else if !(resources <= state.get_bank_resources()) {
                    for resource in Resource::ALL.iter() {
                        if resources[*resource] > state.get_bank_resources()[*resource] {
                            return Err(Error::NoMoreResourceInBank(*resource));
                        }
                    }
                    unreachable!();
                } else {
                    Ok(())
                }
            },
            //
            // ## Use Monopole Development Card
            //
            Action::DevelopmentMonopole { resource: _ } => {
                if !turn_phase.unbound() {
                    Err(Error::IncoherentAction(action))
                } else if *development_phase != DevelopmentPhase::Ready {
                    Err(Error::DevelopmentCardAlreadyPlayed)
                } else if state.get_player_hand(*player).development_cards.monopole == 0 {
                    Err(Error::NoCard { card_type: DevelopmentCard::Monopole })
                } else {
                    Ok(())
                }
            },

            _ => unimplemented!(),
        }
        _ => panic!("Game already finished"),
    }
}
