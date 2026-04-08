use catan::utils::{Coord, Resource, Resources};
use catan::state::PlayerId;

use catan::game::Action;

#[derive(Clone, Debug)]
pub enum ParsingError {
    WrongKeyword(String),
    CouldntParseNumber(String),
    CouldntParseCoord(String),
    CouldntParseResource(String),
    CouldntParsePlayer(String),
    NotEnoughtParameters,
}

pub fn end_parse_number(raw: &str) -> Result<i8, ParsingError> {
    raw.parse::<i8>().map_err(|_| ParsingError::CouldntParseNumber(raw.to_string()))
}

pub fn end_parse_player_id(raw: &str) -> Result<PlayerId, ParsingError> {
    match raw.to_lowercase().as_str() {
        "none" | "n" | "-" => Ok(PlayerId::NONE),
        _ => {
            let id = raw.parse::<u8>()
                .map_err(|_| ParsingError::CouldntParsePlayer(raw.to_string()))?;
            Ok(PlayerId::from(id))
        }
    }
}

pub fn end_parse_optional_player_id(raw: Option<&str>) -> Result<PlayerId, ParsingError> {
    match raw {
        Some(player_str) => end_parse_player_id(player_str),
        None => Ok(PlayerId::NONE), // デフォルトでNONE
    }
}

pub fn end_parse_coord(raw: &str) -> Result<Coord, ParsingError> {
    let mut split_raw = raw.split(",");
    let potential_error = || ParsingError::CouldntParseCoord(raw.to_string());
    let x = split_raw.next().ok_or_else(potential_error)?.parse::<i8>().or_else(|_| Err(potential_error()))?;
    let y = split_raw.next().ok_or_else(potential_error)?.parse::<i8>().or_else(|_| Err(potential_error()))?;
    Ok(Coord::new(x,y))
}

pub fn end_parse_resource(raw: &str) -> Result<Resource, ParsingError> {
    match raw.chars().next() {
        Some('B') => Ok(Resource::Brick),
        Some('L') => Ok(Resource::Lumber),
        Some('O') => Ok(Resource::Ore),
        Some('G') => Ok(Resource::Grain),
        Some('W') => Ok(Resource::Wool),
        _ => Err(ParsingError::CouldntParseResource(raw.to_string())),
    }
}

pub fn parse_action(raw: String) -> Result<Action, ParsingError> {
    let raw = raw.replace("\n", "");
    let mut splited = raw.split(" ");
    match splited.next() {
        Some("EndTurn") | Some("E") => {
            Ok(Action::EndTurn)
        }
        Some("Quit") | Some("Q") => {
            Ok(Action::Exit)
        }
        Some("RollDice") | Some("Roll") | Some("Dice") | Some("RL") => {
            Ok(Action::RollDice)
        }
        Some("BuildRoad") | Some("Road") | Some("R") => {
            let path = end_parse_coord(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            Ok(Action::BuildRoad{ path })
        }
        Some("BuildSettlement") | Some("Settlement") | Some("S") => {
            let intersection = end_parse_coord(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            Ok(Action::BuildSettlement { intersection })
        }
        Some("BuildCity") | Some("City") | Some("C") => {
            let intersection = end_parse_coord(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            Ok(Action::BuildCity { intersection })
        }
        Some("BuyDevelopmentCard") | Some("DevelopmentCard") | Some("Development") | Some("D") => {
            Ok(Action::BuyDevelopment)
        }
        Some("PlayKnight") | Some("Knight") | Some("K") => {
            Ok(Action::DevelopmentKnight)
        }
        Some("PlayRoadBuilding") | Some("RoadBuilding") | Some("RB") => {
            Ok(Action::DevelopmentRoadBuilding)
        }
        Some("PlayYearOfPlenty") | Some("YearOfPlenty") | Some("YP") => {
            let resource1 = end_parse_resource(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let resource2 = end_parse_resource(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let mut resources = Resources::ZERO;
            resources += Resources::new_one(resource1, 1);
            resources += Resources::new_one(resource2, 1);
            Ok(Action::DevelopmentYearOfPlenty { resources })
        }
        Some("PlayMonopole") | Some("Monopole") | Some("M") => {
            let resource = end_parse_resource(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            Ok(Action::DevelopmentMonopole { resource })
        }
        Some("Keep") | Some("KP") => {
            let brick = end_parse_number(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let lumber = end_parse_number(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let ore = end_parse_number(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let grain = end_parse_number(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let wool = end_parse_number(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;

            Ok(Action::Keep { resources: Resources::new(brick, lumber, ore, grain, wool)})
        }
        Some("MoveThief") | Some("Thief") | Some("MT") => {
            let hex = end_parse_coord(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let victim = end_parse_optional_player_id(splited.next())?;
            Ok(Action::MoveThief { hex, victim })
        }
        Some("TradeBank") | Some("Trade") | Some("T") => {
            let given = end_parse_resource(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let asked = end_parse_resource(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            Ok(Action::TradeBank { given, asked })
        }
        Some("TradePlayers") | Some("TP") => {
            let offer_brick = end_parse_number(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let offer_lumber = end_parse_number(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let offer_ore = end_parse_number(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let offer_grain = end_parse_number(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let offer_wool = end_parse_number(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let require_brick = end_parse_number(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let require_lumber = end_parse_number(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let require_ore = end_parse_number(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let require_grain = end_parse_number(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let require_wool = end_parse_number(splited.next().ok_or(ParsingError::NotEnoughtParameters)?)?;
            let trade_partner = end_parse_optional_player_id(splited.next())?;

            let offer = Resources::new(offer_brick, offer_lumber, offer_ore, offer_grain, offer_wool);
            let want = Resources::new(require_brick, require_lumber, require_ore, require_grain, require_wool);

            Ok(Action::TradePlayers { offer, want, partner: trade_partner })
        }
        Some("TradePlayersAccept") | Some("TPA") => {
            Ok(Action::TradePlayersAccept)
        }
        Some("TradePlayersDecline") | Some("TPD") => {
            Ok(Action::TradePlayersDecline)
        }
        Some(other) => {
            Err(ParsingError::WrongKeyword(other.to_string()))
        }
        None => {
            Err(ParsingError::NotEnoughtParameters)
        }
    }
}

pub fn parse_help() -> &'static str {"
Resource: [B]rick [L]umber [O]re [G]rain [W]ool
Coord: <x>,<y>
Action: [E]ndTurn
        [RL] RollDice
        Build[R]oad <Coord> / Build[S]ettlement <Coord> / Build[C]ity <Coord>
        Buy[D]evelopmentCard
        Play[K]night / Play[RB]RoadBuilding / Play[YP]YearOfPlenty <Resource1> <Resource2> / Play[M]onopole <Resource>
        [KP] Keep <Brick> <Lumber> <Ore> <Grain> <Wool> / Move[MT]hief <Coord> [PlayerId / [n]one]
        [T]radeBank <Resource> <Resource>
        [TP]radePlayers <Offer: B L O G W> <Require: B L O G W> <PlayerId>
        [TPA]ccept / [TPD]ecline
        [Q]uit
"}
