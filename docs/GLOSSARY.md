# Glossary

One word per thing, on every page — the globe (`index.html`), the scan gallery (`view.html`)
and the data console (`admin.html`) — and in the code and docs behind them. When a screen
needs a word for one of these, it uses the word here; when the API's value is not a word a
person would say, the screen shows the label in the last column instead (`apps/web/src/lib/labels.ts`,
`lib/format.ts`).

## Places and what is in them

| Word           | Means                                                                                                                                                                                       | Not                                                                                                                                                                                                     |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Site**       | A place on Earth the operator works: a boundary, the 3D model of it, its saved views and the project attached to it. Chosen in the site switcher (top left).                                | "Project" in the UI. A site is shown by **one name**: its project's name when a mission project is attached (the demo site is "Blackrock Mesa"), else its catalog name (`features/sites/siteNames.ts`). |
| **Project**    | The mission data attached to a site: zones, machines, plans, the agent's work. Code and docs only; on screen it is the site.                                                                |                                                                                                                                                                                                         |
| **Zone**       | An area of a site that work is planned for (`Z-21 North fence`). Drawn areas are zones too (`A-01`).                                                                                        |                                                                                                                                                                                                         |
| **Model**      | The site's 3D reality model, in one of three **representations**: **Splat** (Gaussian splat, visual only), **Mesh**, **Points** (point cloud). The strip at a site switches between them.   | The **asset** record or its file name ("Gaussian splat (LOD)", "Cesium Gaussian splat demo") — those are catalog entries, never what the UI calls the site.                                             |
| **Layer**      | Data drawn on the globe that is not a site's model: imagery, terrain, land cover, buildings, GeoJSON. **Favourites** are the four at the top of Layers: Imagery, Vegetation, Zones, Tracks. |                                                                                                                                                                                                         |
| **Saved view** | A camera position kept for a site, in the site switcher (`b`).                                                                                                                              | "Bookmark" on screen (the API's `cameraBookmarks`).                                                                                                                                                     |

## Work

| Word        | Means                                                                                                                                                   | Not                                                     |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **Machine** | A robot in the fleet (`TR-07 Harrier`). The **Fleet** is a site's machines; Fleet opens as the drawer on the right.                                     | "Robot", "unit", "vehicle" on screen.                   |
| **Plan**    | A mission for the fleet: goal, zones, machines, schedule, steps. Drafted by the agent, approved by the operator. Plan opens as the drawer on the right. | "Mission" as a noun on screen, except "Plan a mission". |
| **Agent**   | The assistant behind the command box's last row and the status line's second half. It drafts plans and runs instructions.                               |                                                         |

## Data

| Word        | Means                                                                                                                                                               | Labels shown for API values                                                                                            |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Capture** | What someone recorded and sent: a video, photos, a splat or a point cloud, with its files. Added from **Add › Upload a capture** or the phone page (`upload.html`). | Kind: Video, Photos, Splat, Point cloud. Status: Waiting for files, Ready, Processing, Done, Failed, Cancelled.        |
| **Run**     | One pass of the pipeline over a capture (a recipe with its parameters, on a provider); made of **stages**. The data console lists runs; the API calls them jobs.    | Status: Queued, Running, Done, Failed, Cancelled ("not-started" is Queued).                                            |
| **Output**  | A file a run wrote (an artifact): frames, poses, a splat, 3D Tiles, a manifest…                                                                                     | Kind: Frames, Camera poses, Masks, Splat, Mesh, Points, 3D Tiles, Thumbnail, Ground samples, Manifest, Clip, Metadata. |
| **Scan**    | A capture that became a site with a splat, as the scan gallery (`view.html`) shows it: one finished model on its own, to turn over on a phone.                      |                                                                                                                        |
| **Source**  | Data that is already hosted (a Cesium ion asset, a 3D Tiles URL, GeoJSON, an imagery service, STAC) and is linked rather than uploaded: **Add › Link a source**.    |                                                                                                                        |

## Where things are

| Surface           | What it is                                                                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Site switcher** | Top left: the site's name; its menu lists sites, the site's saved views, Add a site, and the way to the scan gallery and the data console. |
| **Command box**   | Top centre (`⌘K`, `/`): search, every action, and the agent. On a phone, the search button opens it full screen.                           |
| **Tools**         | The left rail: Layers, Measure, Add, Settings. On a phone they are under More.                                                             |
| **Drawer**        | Plan or Fleet, on the right edge (a bottom sheet on a phone). The map beside it stays live.                                                |
| **Status line**   | Bottom left: the fleet in view and the agent's line; its chevron opens the activity log.                                                   |
| **Site pin**      | What a site's markers and zone chips collapse into above 50 km: the site's name and how many things it stands for.                         |
