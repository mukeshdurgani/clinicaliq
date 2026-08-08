"""
mcp_server.py
--------------
Standalone MCP server exposing ClinicalIQ's two database tools --
query_doctor and query_service -- over the MCP protocol (US-06 Part 1).

Self-contained on purpose: this file does not import from the clinicaliq
package so it can be run and inspected in isolation, exactly like a real
MCP server would be deployed. The SQL logic mirrors clinicaliq/tools.py's
original @tool functions -- only the decorator changes (@tool -> @mcp.tool()).

Run standalone:
    python s01/mcp_server.py

Inspect with MCP Inspector:
    npx @modelcontextprotocol/inspector python s01/mcp_server.py
    Open http://localhost:5173 -- both tools should appear.
"""
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("clinicaliq-tools")

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH  = DATA_DIR / "clinic_data.db"

# 3-letter day abbreviations match the doctors.available_days column format
# (e.g. "Mon,Wed,Fri") seeded by data/seed.py -- normalise before querying so
# "Friday", "friday", and "Fri" all match the same row.
_DAY_ABBREV = {
    "monday": "Mon", "tuesday": "Tue", "wednesday": "Wed", "thursday": "Thu",
    "friday": "Fri", "saturday": "Sat", "sunday": "Sun",
}


def _normalise_day(day: str) -> str:
    return _DAY_ABBREV.get(day.strip().lower(), day.strip()[:3].title())


# Apollo Health Clinic's 10 departments (per clinicaliq/config.py SYSTEM_PROMPT)
# mapped to the singular noun for a practitioner in that specialty -- used only
# to phrase the "no doctor listed" message naturally (e.g. "dermatologist",
# not "dermatologys").
_SPECIALTY_NOUN = {
    "cardiology": "cardiologist", "orthopaedics": "orthopaedist",
    "dermatology": "dermatologist", "gynaecology": "gynaecologist",
    "paediatrics": "paediatrician", "ent": "ENT specialist",
    "ophthalmology": "ophthalmologist", "neurology": "neurologist",
    "general medicine": "General Medicine doctor", "dental": "dentist",
}


def _no_specialty_match(specialty: str) -> str:
    noun = _SPECIALTY_NOUN.get(specialty.strip().lower(), f"{specialty} specialist")
    return (
        f"No {noun}s are currently listed at Apollo Health Clinic. "
        "Please call reception for specialist referrals."
    )


@mcp.tool()
def query_doctor(specialty: str | None = None, day: str | None = None, name: str | None = None) -> str:
    """Fetch current doctor availability and consultation fees from the database.

    Args:
        specialty: Department/specialty to filter by, e.g. "Cardiology" (optional).
        day: Day of the week to filter by, e.g. "Friday" or "Fri" (optional).
        name: Doctor name (or partial name) to filter by, e.g. "Meera Nair" (optional).

    Returns formatted doctor availability as a plain-text string, or a
    structured "not found" message if no doctor matches -- never invents a
    doctor, schedule, or fee not present in the database.
    """
    # check_same_thread=False: SQLite normally raises ProgrammingError if a connection
    # is used from a thread other than the one that created it. The MCP server may
    # dispatch a tool call on a different thread than the one that opened the
    # connection. This flag disables that check.
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)

    query  = "SELECT name, specialty, available_days, consultation_fee FROM doctors WHERE 1=1"
    params = []
    if specialty:
        query += " AND specialty LIKE ?"
        params.append(f"%{specialty}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if day:
        query += " AND available_days LIKE ?"
        params.append(f"%{_normalise_day(day)}%")
    query += " ORDER BY specialty, name"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        if specialty:
            return _no_specialty_match(specialty)
        return "No doctors matched that search. Please call reception to confirm availability."

    return "\n".join(
        f"{doctor_name} ({doctor_specialty}): available {available_days} | Consultation fee Rs. {fee}"
        for doctor_name, doctor_specialty, available_days, fee in rows
    )


@mcp.tool()
def query_service(service_name: str) -> str:
    """Fetch current pricing for a clinic service or health package from the database.

    Args:
        service_name: Name or id of the service/package, e.g. "ECG", "ecg",
            "Full Body Checkup", or "comprehensive".

    Returns formatted price/duration information as a plain-text string, or a
    structured "not found" message if nothing matches -- never invents a
    service, package, or price not present in the database.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)  # see query_doctor for why

    services = conn.execute(
        "SELECT name, department, average_duration_mins, price FROM services "
        "WHERE service_id LIKE ? OR name LIKE ?",
        (f"%{service_name}%", f"%{service_name}%"),
    ).fetchall()
    if services:
        conn.close()
        return "\n".join(
            f"{name} ({department}): {duration} min | Rs. {price}"
            for name, department, duration, price in services
        )

    packages = conn.execute(
        "SELECT name, tests_included, price, recommended_for FROM health_packages "
        "WHERE package_id LIKE ? OR name LIKE ?",
        (f"%{service_name}%", f"%{service_name}%"),
    ).fetchall()
    conn.close()
    if packages:
        return "\n".join(
            f"{name}: {tests_included} | Rs. {price} (recommended for {recommended_for})"
            for name, tests_included, price, recommended_for in packages
        )

    return (
        f"No service or health package matching '{service_name}' was found. "
        "Please call reception to confirm current offerings."
    )


if __name__ == "__main__":
    mcp.run()  # STDIO transport by default
