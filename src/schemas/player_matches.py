from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class PlayerView:
    shot_id: str
    track_id: str


@dataclass
class MatchedPlayer:
    player_id: str
    team: str
    views: list[PlayerView] = field(default_factory=list)


@dataclass
class PlayerMatches:
    matched_players: list[MatchedPlayer] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PlayerMatches":
        data = json.loads(path.read_text())
        players = []
        for p in data.pop("matched_players"):
            views = [PlayerView(**v) for v in p.pop("views")]
            players.append(MatchedPlayer(views=views, **p))
        return cls(matched_players=players)
