from football_odds_scraper.team_colors import get_team_color, TEAM_COLORS


def test_team_colors_has_all_world_cup_teams():
    from football_odds_scraper.world_cup_teams import WORLD_CUP_TEAMS

    for team in WORLD_CUP_TEAMS.values():
        assert team["name"] in TEAM_COLORS


def test_get_team_color_belgium():
    assert get_team_color(4717) == "#EF3340"
