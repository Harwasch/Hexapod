# Public parcel lookup

The landing page's "Find my land" (`finder.html`) asks `POST /api/v1/public/parcels/near`: county
assessor parcels near a point, for a visitor who has not signed in. Code:
`apps/api/app/api/v1/public.py`, `apps/api/app/services/public_parcels.py`, and the county lists in
`apps/api/app/research/providers/parcels.py`.

## Which counties it asks

**Verified counties only** (Harrison's decision, 2026-10-20). Every available California county's
parcel layer, service and ArcGIS item metadata was read on 2026-10-20. The lookup asks a county only
if its published terms were read and they allow showing the outlines with attribution: no terms
stated, or disclaimers only (no warranty, not a survey). 40 counties qualify (`PUBLIC_VERIFIED`).

Kern is one of them, but its terms also require its disclaimer statement with any copy of its data.
The lookup returns the full statement as a notice, and the finder and the form show it. The statement
also makes the recipient indemnify the data providers, which is an open question for Harrison.

The signed-in app's parcel picker is unchanged: it still asks every available county.

## Excluded, and why

| County                                                                                                                          | Why                                                                                                                                                                        | Until                                      |
| ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| El Dorado (06017)                                                                                                               | Its item licence: "These results are not to be redistributed unless agreed upon with the originator."                                                                      | An agreement with the county               |
| Sonoma (06097)                                                                                                                  | Its licence allows sharing with the attribution "County of Sonoma", but not redistributing modified data. Whether showing chosen parcels together counts needs confirming. | Its terms are confirmed for this use       |
| Humboldt, Los Angeles, Madera, Marin, Mono, Monterey, Napa, Orange, Sacramento, San Diego, San Luis Obispo, Santa Cruz, Ventura | Terms could not be read: no reachable ArcGIS item, and the layer and service metadata state none.                                                                          | Each county's terms are read and confirmed |

Not used anywhere, public or signed in: Sierra (its data is sold), Modoc and Plumas (no online parcel
service).

## Enabling a county

Once a county's terms are confirmed for public display, move its FIPS code from `PENDING_TERMS` (or
`NO_PUBLIC_REUSE`) to `PUBLIC_VERIFIED` in `parcels.py`. Record what the terms say in the comment
there and in this file.

- If the terms require a statement with the data (as Kern's do), add it to the county's
  `disclaimer` in `apps/api/app/research/providers/data/ca_parcels.json`, and the lookup will return
  it.
- `tests/test_public_parcels.py` checks that every available county is in exactly one of the three
  lists. Its assertions on the excluded set change with the move.

## Limits

- **Search radius:** at most 300 m.
- **Per client:** 8 lookups at once, then 12 a minute.
- **County timeout:** 7 s per county.
- **Cache:** answers kept 10 minutes, keyed by the point rounded to about 11 m.
- **Concurrency:** at most 4 lookups at once across all visitors (503 beyond that).
- **Request body:** at most 4 KB.

All of these are in-process and per Fly machine; `public_parcels.py` and `ratelimit.py` say why.
