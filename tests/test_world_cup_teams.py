from football_odds_scraper.world_cup_teams import (
    WORLD_CUP_TEAMS,
    country_code_to_flag,
    get_team_display,
)


def test_country_code_to_flag():
    assert country_code_to_flag("AR") == "🇦🇷"
    assert country_code_to_flag("mx") == "🇲🇽"
    assert country_code_to_flag("") == "🏳️"


def test_get_team_display_known_team():
    assert get_team_display(4819) == "🇦🇷 Argentina"
    assert get_team_display(4781) == "🇲🇽 Mexico"


def test_get_team_display_gb_teams():
    assert get_team_display(4713) == "🇬🇧 England"
    assert get_team_display(4695) == "🇬🇧 Scotland"


def test_get_team_display_unknown_team():
    assert get_team_display(99999) == "Equipo 99999"


def test_world_cup_teams_has_48_entries():
    assert len(WORLD_CUP_TEAMS) == 48
