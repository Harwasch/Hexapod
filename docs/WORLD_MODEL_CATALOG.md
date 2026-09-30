# World model catalog: explicit 3D and 4D state

A broad sweep written 2026-09-30 on branch `living-models`. It extends
[GAUSSIAN_WORLD_MODELS.md](GAUSSIAN_WORLD_MODELS.md) and
[WORLD_MODEL_ROUTES.md](WORLD_MODEL_ROUTES.md), whose earlier survey sampled about 25 works.
This sweep covers roughly 150.

Sources:

- the curated lists worldbench/awesome-3d-4d-world-models (survey arXiv 2509.07996, TPAMI 26),
  Li-Zn-H/AwesomeWorldModels, knightnemo/Awesome-World-Models, LMD0311/Awesome-World-Model,
  hzxie/Awesome-3D-Scene-Generation and cwchenwang/awesome-4d-generation;
- Hugging Face paper search and arXiv;
- company announcements.

Most entries were checked only as far as a list, a search result or an abstract. **Nothing
here has had its licence checked unless the row says so.** Treat the rest as UNVERIFIED.

## What the sweep changed

1. **The field is bigger and faster than the first survey showed.** 2026 brought a wave of
   Gaussian world models for manipulation (GaussianDream++, GaussianWAM, PhysMani,
   ContactGaussian-WM, RoGSW4RLD, DyG²T) and dozens of occupancy world models for driving.
2. **Still none at outdoor-site scale with action-conditioned dynamics.** The Gaussian-state
   models are tabletop or object scale. The outdoor ones are driving occupancy or LiDAR over
   seconds. That conclusion stands.
3. **Three new things matter for Hexapod:**
   - **native-3DGS geospatial generation**: ABot-Earth 0.5, MetaEarth3D, Skyfall-GS, Sat2City;
   - **commercial splat-generation APIs**: SpAItial Echo-2;
   - **video world models with an explicit 3D memory**: Spatia, EvoWorld, SPMem, Mem-World,
     LSM-World. These are a grounded version of the original "video model → views → Gaussians"
     idea, where our measured splats _are_ the memory.
4. **The best template with code and weights is PointWorld (NVIDIA, CVPR 26).** It takes a
   point cloud plus robot actions expressed as 3D points and predicts 3D point flow for the whole
   scene. It was trained on about 500 h of data. Its weights licence is UNVERIFIED.
5. **Applied Intuition's Neural Sim** is a commercial proof of the dream-mode shape: a 3DGS
   reconstruction plus animated actors plus closed-loop what-ifs, for road scenes.

## By view

| View       | Most relevant                                                                                                                                                                                                                                                                                                                                                                       |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Static** | **ABot-Earth 0.5** (2606.09967: satellite → native, georeferenced 3DGS, under 10 min/km², 300+ cities; licence unclear) as context _around_ a scan · **SpAItial Echo-2** (REST API, SPZ/SOG output, ~$1.60–8 per generation) · FlashWorld (ICLR 26), Bolt3D, GaussianCity · Skyfall-GS, MetaEarth3D, Sat2City v2                                                                    |
| **Living** | PhysMani (divergence-free Gaussian velocity field, code) · GaussianFluent, PhysGM (physics parameters on Gaussians) · 4D capture products (Gracia, 4DV.ai, Volinga) for recorded replay · Spark 2.0 (World Labs, open source) for streaming more than 100 M splats                                                                                                                  |
| **Live**   | Embodied Gaussians / PEGS (BDAI, CoRL 24: a simulator corrected by vision in real time, code) · Spatia (a static point cloud updated by SLAM, dynamics separate) · GaussTwin · GaussianWorld · Niantic Spatial VPS and Large Geospatial Model for localisation · Vantor WorldView 3D (a global 3D base refreshed daily)                                                             |
| **Dream**  | **PointWorld** (action-conditioned 3D flow, code and weights) · GS-Dynamics, ParticleFormer, PIN-WM, DreMa (Gaussians plus learned or physical dynamics) · PerpetualWonder (CVPR 26, action-conditioned 4D) · RoGSW4RLD (lifts action-conditioned video rollouts into metric 4DGS) · Applied Intuition Neural Sim, Waabi World · GaussGym (3DGS inside IsaacGym, for legged robots) |

## Catalog

### A. Gaussian or particle state world models (actions or dynamics)

GWM · ManiGaussian / ++ (code) · MRO-GWM · GaussianDream / ++ (20 Gaussian tokens in a VLA) ·
GaussianWAM · GAF (Gaussian Action Field) · PhysMani (code) · ContactGaussian-WM · PEGS / Embodied
Gaussians (code) · GS-Dynamics (code) · DyG²T · DSG-World · DreMa · PIN-WM (code) · PhysTwin ·
PGND · GausSim · 3DGSim · ParticleGS · ParticleFormer · DeformMaster · PhysGM · GaussianFluent ·
LagrangeGS / VersaGauss · GaussTwin · RoboGSim · Real2Sim-Eval (code) · GaussGym · RoGSW4RLD ·
RynnWorld-4D (code) · GEM-4D · ORV (code) · WristWorld (code) · Puffin-World · TourPhysics ·
EgoPhys · PointWorld (code and weights)

### B. Occupancy, LiDAR and Gaussian world models for driving

OccWorld · DOME · OccSora · OccLLaMA · Drive-OccWorld · PreWorld · OccProphet · RenderWorld ·
DIO · T³Former · COME · I²-World · OccTENS · SparseWorld · OCCVAR · DFIT-OccWorld · CascadeOcc ·
OWMDrive · ForecastOcc · InterOCF · VISA · OccSim · OccDirector · AutoWorld · GaussianAD ·
DriveWorld · UniWorld · UnO · Cam4DOcc · UniOcc · GenieDrive · UniScene · Copilot4D · ViDAR ·
BEVWorld · HERMES / ++ · LidarDM · LiDARCrafter · LiSTAR · U4D · GEM (LiDAR) · OpenDWM ·
DriveX · GaussianWorld · GEM (4D Gaussian) · 4DGS-WAM · GaussianDWM / ++ · WorldSplat ·
DiST-4D · UniFuture · DriveDreamer4D · DrivingRecon · Envision4D · Xiaomi Auto World Model

### C. Video world models with explicit 3D memory

GEN3C · Spatia (code) · EvoWorld (code) · SPMem · Mem-World · LSM-World / Mirage (a latent 3D
cache, 55× less GPU memory than an RGB cache) · WorldWeave · VerseCrafter · Memory Forcing ·
WorldPlay · PanoWorld · HunyuanWorld-Voyager (RGB-D plus a point-cloud cache) · MoVerse
(conditioned on Gaussian renders)

### D. Explicit 3D and 4D world generators

- **Static 3DGS:** FlashWorld · Bolt3D · Wonderland · WonderTurbo · WonderZoom · SplatFlow ·
  VideoRFSplat · LucidDreamer · DreamScene360 · FlexWorld · Scene Splatter · PixWorld ·
  GaussianCity · Proc-GS · Lyra · Apple SHARP
- **Mesh, point and panorama:** HunyuanWorld 1.0 / HY-World 2.0 · Matrix-3D · Meta WorldGen ·
  Terra · WorldExplorer · SceneFoundry · EmbodiedGen · SynCity · UrbanWorld · TRELLIS.2 · Seed3D
- **Voxel and occupancy:** XCube · SCube · InfiniCube · SemCity · X-Scene · DrivingSphere ·
  DynamicCity (4D)
- **4D:** PerpetualWonder · CityDreamer4D · Diff4Splat · CAT4D · 4Real · GenXD · DimensionX ·
  Free4D · 4K4DGen · DeepVerse · NeoVerse · Code2Worlds · INSPATIO-WORLD
- **Geospatial:** ABot-Earth 0.5 · MetaEarth3D · Skyfall-GS · Sat2City v2 · Orbit-to-Ground ·
  GS-Voxel

### E. Companies and products (splat-native or 3D)

| Company                 | Product                                     | Output                      | Dynamics          | Geo                  | Access                    |
| ----------------------- | ------------------------------------------- | --------------------------- | ----------------- | -------------------- | ------------------------- |
| AMAP (Alibaba)          | ABot-Earth 0.5                              | native 3DGS from satellite  | static            | **yes**, 300+ cities | platform; licence unclear |
| SpAItial                | Echo-2                                      | 3DGS (SPZ/SOG) + mesh       | static            | no                   | **REST API + MCP**        |
| Niantic Spatial         | Large Geospatial Model, Scaniverse, VPS 2.0 | splats, mesh, VPS maps      | localisation      | **yes**              | enterprise SDK            |
| World Labs              | Marble, Atlas, Spark 2.0                    | splats; LOD streaming       | none shown        | no                   | API; Spark open source    |
| Applied Intuition       | Neural Sim                                  | 3DGS + animated actors      | **closed loop**   | roads                | enterprise                |
| Waabi                   | Waabi World                                 | 3DGS / 4D                   | yes               | driving              | internal                  |
| Bentley / Cesium        | ion reality modelling                       | splat 3D Tiles with LOD     | —                 | **yes**              | commercial                |
| Esri                    | ArcGIS GaussianSplatLayer                   | georeferenced splats        | —                 | **yes**              | commercial                |
| Vantor (ex-Maxar)       | WorldView 3D                                | mesh / DSM, refreshed daily | refresh           | **yes**, global      | commercial                |
| Duality AI              | Falcon Site Twins                           | UE5 twins from GIS data     | sensor simulation | **yes**              | commercial                |
| Gracia, 4DV.ai, Volinga | 4DGS capture and streaming                  | 4D splats                   | recorded          | no                   | tools / plugins           |
| Google                  | Genie 3 + Street View                       | video                       | actions           | real streets         | consumer only             |
| Odyssey, Decart, Runway | Odyssey-3, Oasis 3, GWM-1                   | video                       | actions           | no                   | APIs                      |
| Microsoft, Meta, Apple  | TRELLIS.2 (MIT), SAM 3D, WorldGen, SHARP    | mesh / 3DGS                 | —                 | no                   | open weights (some)       |
