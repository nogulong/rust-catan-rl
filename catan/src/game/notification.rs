use crate::utils::Resources;
use crate::game::Action;
use crate::state::PlayerId;

#[derive(Clone, Debug, PartialEq)]

pub enum Notification {
    ActionPlayed {
        by: PlayerId,
        action: Action,
    },
    ResourcesRolled {
        roll: u8,
        resources: Vec<Resources>,
    },
    Discards {// まとめて通知
        discards: Vec<(PlayerId, Option<Resources>)>,
    },
    TradeAccepted,
    TradeDeclined,
    GameFinished {
        winner: PlayerId,
    },
    ThiefRolled,
    InitialPlacementFinished,
}
