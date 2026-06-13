"""Mapeo estático OddsPapi participantId → equipo del Mundial 2026."""

from __future__ import annotations

WORLD_CUP_TEAMS: dict[int, dict[str, str]] = {
    4735: {"name": "Korea Republic", "code": "KR"},
    4714: {"name": "Czechia", "code": "CZ"},
    4752: {"name": "Canada", "code": "CA"},
    4479: {"name": "Bosnia and Herzegovina", "code": "BA"},
    4724: {"name": "USA", "code": "US"},
    4789: {"name": "Paraguay", "code": "PY"},
    4792: {"name": "Qatar", "code": "QA"},
    4699: {"name": "Switzerland", "code": "CH"},
    4748: {"name": "Brazil", "code": "BR"},
    4778: {"name": "Morocco", "code": "MA"},
    7229: {"name": "Haiti", "code": "HT"},
    4695: {"name": "Scotland", "code": "GB"},
    4741: {"name": "Australia", "code": "AU"},
    4700: {"name": "Turkiye", "code": "TR"},
    4711: {"name": "Germany", "code": "DE"},
    55827: {"name": "Curacao", "code": "CW"},
    4705: {"name": "Netherlands", "code": "NL"},
    4770: {"name": "Japan", "code": "JP"},
    4768: {"name": "Ivory Coast", "code": "CI"},
    4757: {"name": "Ecuador", "code": "EC"},
    4688: {"name": "Sweden", "code": "SE"},
    4729: {"name": "Tunisia", "code": "TN"},
    4698: {"name": "Spain", "code": "ES"},
    4753: {"name": "Cape Verde", "code": "CV"},
    4717: {"name": "Belgium", "code": "BE"},
    4758: {"name": "Egypt", "code": "EG"},
    4834: {"name": "Saudi Arabia", "code": "SA"},
    4725: {"name": "Uruguay", "code": "UY"},
    4766: {"name": "IR Iran", "code": "IR"},
    4784: {"name": "New Zealand", "code": "NZ"},
    4481: {"name": "France", "code": "FR"},
    4739: {"name": "Senegal", "code": "SN"},
    4767: {"name": "Iraq", "code": "IQ"},
    4475: {"name": "Norway", "code": "NO"},
    4819: {"name": "Argentina", "code": "AR"},
    4691: {"name": "Algeria", "code": "DZ"},
    4718: {"name": "Austria", "code": "AT"},
    4771: {"name": "Jordan", "code": "JO"},
    4704: {"name": "Portugal", "code": "PT"},
    4823: {"name": "Congo DR", "code": "CD"},
    4713: {"name": "England", "code": "GB"},
    4715: {"name": "Croatia", "code": "HR"},
    4764: {"name": "Ghana", "code": "GH"},
    5164: {"name": "Panama", "code": "PA"},
    4723: {"name": "Uzbekistan", "code": "UZ"},
    4820: {"name": "Colombia", "code": "CO"},
    4781: {"name": "Mexico", "code": "MX"},
    4736: {"name": "South Africa", "code": "ZA"},
}


def country_code_to_flag(code: str) -> str:
    if not code or len(code) != 2:
        return "🏳️"
    return chr(ord(code[0].upper()) + 127397) + chr(ord(code[1].upper()) + 127397)


def get_team_display(participant_id: int | None) -> str:
    if participant_id is None:
        return "?"
    team = WORLD_CUP_TEAMS.get(int(participant_id))
    if not team:
        return f"Equipo {participant_id}"
    flag = country_code_to_flag(team.get("code", ""))
    return f"{flag} {team['name']}"
