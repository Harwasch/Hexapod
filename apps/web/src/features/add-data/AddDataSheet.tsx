import {
  Box,
  ChevronRight,
  FileJson,
  Globe2,
  Image,
  Layers3,
  MapPinned,
  Upload,
  UploadCloud,
} from "lucide-react";
import { useState, type ReactNode, type SubmitEvent } from "react";

import type { AssetInput, LayerCreate, Representation, SiteCreate } from "@twin/contracts";
import { circleFootprint, footprintAreaM2, formatArea } from "@twin/geo";
import {
  GlassButton,
  GlassField,
  GlassInput,
  GlassSegmentedControl,
  GlassSelect,
  GlassSheet,
  GlassSwitch,
  GlassTextarea,
  useFieldId,
} from "@twin/ui";

import {
  useCreateLayer,
  useCreateSite,
  useIonStatus,
  useSites as useSiteCatalog,
} from "@/api/queries";
import { ApiError } from "@/api/client";
import { useScene } from "@/cesium/SceneContext";
import { describeError } from "@/lib/log";
import {
  parseFootprintText,
  validateAssetId,
  validateDate,
  validateHttpUrl,
  validateName,
} from "@/lib/validation";
import { useSettings } from "@/state/settings";
import { useToasts } from "@/state/toasts";
import { useUi } from "@/state/ui";

type Tab = "site" | "ion" | "tiles" | "geojson" | "imagery" | "stac";

const TABS: { id: Tab; label: string; icon: typeof Box }[] = [
  { id: "site", label: "Reality model", icon: MapPinned },
  { id: "ion", label: "ion asset", icon: Box },
  { id: "tiles", label: "3D Tiles URL", icon: Layers3 },
  { id: "geojson", label: "GeoJSON", icon: FileJson },
  { id: "imagery", label: "Imagery service", icon: Image },
  { id: "stac", label: "STAC", icon: Globe2 },
];

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    const fields = error.fieldErrors;
    return fields.length ? fields.join("; ") : error.message;
  }
  return describeError(error);
}

export function AddDataSheet() {
  const open = useUi((s) => s.addDataOpen);
  const setOpen = useUi((s) => s.setAddDataOpen);
  const [tab, setTab] = useState<Tab>("site");
  const sites = useSiteCatalog();
  const setPanel = useUi((s) => s.setPanel);
  return (
    <GlassSheet
      open={open}
      onOpenChange={setOpen}
      title="Add data"
      description={
        sites.builtin
          ? "Offline: nothing can be saved right now."
          : "Link a model or map source that is already hosted."
      }
      side="right"
      testId="add-data"
    >
      {/* Most people arriving here have files, not URLs: send them to the uploader first. */}
      <button
        type="button"
        className="add-upload"
        data-testid="add-data-upload"
        onClick={() => {
          setOpen(false);
          setPanel("captures");
        }}
      >
        <UploadCloud size={20} aria-hidden="true" />
        <span className="add-upload__text">
          <strong>Upload a video, photos or a splat</strong>
          <span>Videos and photos become 3D models; splats are placed as they are.</span>
        </span>
        <ChevronRight size={16} aria-hidden="true" />
      </button>
      <p className="panel__eyebrow">Or link a hosted source</p>
      <div className="tabs" role="tablist" aria-label="Data type">
        {TABS.map(({ id, label, icon: Icon }) => (
          <GlassButton
            key={id}
            size="sm"
            role="tab"
            aria-selected={tab === id}
            active={tab === id}
            onClick={() => setTab(id)}
            leadingIcon={<Icon size={14} aria-hidden="true" />}
            data-testid={`add-tab-${id}`}
          >
            {label}
          </GlassButton>
        ))}
      </div>
      <div role="tabpanel">
        {tab === "site" && <SiteForm onDone={() => setOpen(false)} disabled={sites.builtin} />}
        {tab === "ion" && (
          <LayerForm kind="ion" onDone={() => setOpen(false)} disabled={sites.builtin} />
        )}
        {tab === "tiles" && (
          <LayerForm kind="tiles" onDone={() => setOpen(false)} disabled={sites.builtin} />
        )}
        {tab === "geojson" && (
          <LayerForm kind="geojson" onDone={() => setOpen(false)} disabled={sites.builtin} />
        )}
        {tab === "imagery" && (
          <LayerForm kind="imagery" onDone={() => setOpen(false)} disabled={sites.builtin} />
        )}
        {tab === "stac" && (
          <LayerForm kind="stac" onDone={() => setOpen(false)} disabled={sites.builtin} />
        )}
      </div>
    </GlassSheet>
  );
}

/* ------------------------------------------------------------------------- */

function FootprintField({
  value,
  onChange,
  error,
}: {
  value: string;
  onChange: (v: string) => void;
  error?: string;
}) {
  const id = useFieldId("footprint");
  const scene = useScene();
  const parsed = parseFootprintText(value);
  const units = useSettings((s) => s.units);
  const useCamera = () => {
    const pose = scene?.camera.lastKnownPose ?? scene?.camera.pose();
    if (!pose) return;
    onChange(
      JSON.stringify(
        circleFootprint(
          { longitude: pose.longitude, latitude: pose.latitude },
          Math.max(30, Math.min(2000, pose.altitude * 0.6)),
          24,
        ),
      ),
    );
  };
  const onFile = (file: File | undefined) => {
    if (!file) return;
    file
      .text()
      .then(onChange)
      .catch((err: unknown) => onChange(`Could not read file: ${describeError(err)}`));
  };
  return (
    <GlassField
      label="Footprint (GeoJSON Polygon)"
      htmlFor={id}
      hint={
        parsed.ok
          ? `Area ${formatArea(footprintAreaM2(parsed.value), units)}. Terrain and the global 3D world are clipped inside it.`
          : "Paste a Polygon / MultiPolygon, upload a .geojson, or use the current camera location."
      }
      error={error}
    >
      <GlassTextarea
        id={id}
        mono
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder='{"type":"Polygon","coordinates":[[[lon,lat],…]]}'
        rows={4}
        invalid={Boolean(error)}
        data-testid="footprint-input"
      />
      <div className="glass-row">
        <label className="glass-button glass-button--sm" style={{ cursor: "pointer" }}>
          <Upload size={14} aria-hidden="true" /> Upload .geojson
          <input
            type="file"
            accept=".json,.geojson,application/geo+json,application/json"
            className="sr-only"
            onChange={(e) => onFile(e.target.files?.[0])}
          />
        </label>
        <GlassButton size="sm" variant="ghost" onClick={useCamera} disabled={!scene}>
          Around camera
        </GlassButton>
      </div>
    </GlassField>
  );
}

function SourceFields({
  sourceType,
  onSourceType,
  assetId,
  onAssetId,
  url,
  onUrl,
  errors,
}: {
  sourceType: "cesium-ion" | "3d-tiles-url";
  onSourceType: (v: "cesium-ion" | "3d-tiles-url") => void;
  assetId: string;
  onAssetId: (v: string) => void;
  url: string;
  onUrl: (v: string) => void;
  errors: Record<string, string>;
}) {
  const assetFieldId = useFieldId("asset-id");
  const urlFieldId = useFieldId("tiles-url");
  return (
    <>
      <GlassSegmentedControl
        aria-label="Asset source"
        block
        value={sourceType}
        onValueChange={onSourceType}
        options={[
          { value: "cesium-ion", label: "Cesium ion asset ID" },
          { value: "3d-tiles-url", label: "3D Tiles URL" },
        ]}
      />
      {sourceType === "cesium-ion" ? (
        <GlassField
          label="Cesium ion asset ID"
          htmlFor={assetFieldId}
          error={errors.assetId}
          hint="Tiled in Cesium ion (photo reconstruction outputs: mesh, point cloud, Gaussian splats)."
        >
          <GlassInput
            id={assetFieldId}
            mono
            inputMode="numeric"
            value={assetId}
            onChange={(e) => onAssetId(e.target.value)}
            placeholder="4547222"
            invalid={Boolean(errors.assetId)}
            data-testid="asset-id-input"
          />
        </GlassField>
      ) : (
        <GlassField
          label="tileset.json URL"
          htmlFor={urlFieldId}
          error={errors.url}
          hint="Any valid 3D Tiles 1.0/1.1 tileset served over https with CORS."
        >
          <GlassInput
            id={urlFieldId}
            mono
            value={url}
            onChange={(e) => onUrl(e.target.value)}
            placeholder="https://…/tileset.json"
            invalid={Boolean(errors.url)}
            data-testid="tiles-url-input"
          />
        </GlassField>
      )}
    </>
  );
}

function SiteForm({ onDone, disabled }: { onDone: () => void; disabled: boolean }) {
  const create = useCreateSite();
  const scene = useScene();
  const ion = useIonStatus();
  const push = useToasts((s) => s.push);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [footprint, setFootprint] = useState("");
  const [representation, setRepresentation] = useState<Representation>("gaussian-splat");
  const [sourceType, setSourceType] = useState<"cesium-ion" | "3d-tiles-url">("cesium-ion");
  const [assetId, setAssetId] = useState("");
  const [url, setUrl] = useState("");
  const [observedAt, setObservedAt] = useState("");
  const [sourceOrg, setSourceOrg] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [license, setLicense] = useState("");
  const [attribution, setAttribution] = useState("");
  const [sse, setSse] = useState("16");
  const [clipsWorld, setClipsWorld] = useState(true);
  const [bookmarkFromCamera, setBookmarkFromCamera] = useState(true);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const clearError = (field: string) =>
    setErrors((e) =>
      e[field] ? Object.fromEntries(Object.entries(e).filter(([k]) => k !== field)) : e,
    );
  const nameId = useFieldId("site-name");
  const descId = useFieldId("site-desc");
  const dateId = useFieldId("observed");
  const orgId = useFieldId("org");
  const srcUrlId = useFieldId("src-url");
  const licenseId = useFieldId("license");
  const attrId = useFieldId("attr");
  const sseId = useFieldId("sse");

  const submit = async (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    const nextErrors: Record<string, string> = {};
    const vName = validateName(name);
    if (!vName.ok) nextErrors.name = vName.error;
    const vFootprint = parseFootprintText(footprint);
    if (!vFootprint.ok) nextErrors.footprint = vFootprint.error;
    let source: AssetInput["source"] | null = null;
    if (sourceType === "cesium-ion") {
      const v = validateAssetId(assetId);
      if (!v.ok) nextErrors.assetId = v.error;
      else source = { type: "cesium-ion", assetId: v.value };
    } else {
      const v = validateHttpUrl(url, { endsWith: [".json"] });
      if (!v.ok) nextErrors.url = v.error;
      else source = { type: "3d-tiles-url", url: v.value };
    }
    const vDate = validateDate(observedAt);
    if (!vDate.ok) nextErrors.observedAt = vDate.error;
    const sseValue = Number(sse);
    if (!(sseValue > 0 && sseValue <= 512))
      nextErrors.sse = "Screen-space error must be between 1 and 512.";
    if (sourceUrl.trim()) {
      const v = validateHttpUrl(sourceUrl);
      if (!v.ok) nextErrors.sourceUrl = v.error;
    }
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0 || !vName.ok || !vFootprint.ok || !source || !vDate.ok)
      return;
    const attributionList = attribution.trim()
      ? [
          {
            text: attribution.trim(),
            organization: sourceOrg.trim() || null,
            url: sourceUrl.trim() || null,
          },
        ]
      : [];
    const pose = scene?.camera.lastKnownPose ?? scene?.camera.pose();
    const body: SiteCreate = {
      name: vName.value,
      description: description.trim() || null,
      boundary: vFootprint.value,
      attribution: attributionList,
      license: license.trim() ? { name: license.trim(), requiresAttribution: true } : null,
      assets: [
        {
          name: `${vName.value} ${representation}`,
          representation,
          source,
          footprint: vFootprint.value,
          observedAt: vDate.value,
          attribution: attributionList,
          provenance:
            sourceOrg.trim() || sourceUrl.trim()
              ? {
                  sourceOrganization: sourceOrg.trim() || null,
                  sourceUrl: sourceUrl.trim() || null,
                }
              : null,
          renderConfig: {
            maximumScreenSpaceError: sseValue,
            clipsWorld,
            clipFootprint: "catalog",
            heightOffsetM: 0,
          },
          defaultVisible: true,
        },
      ],
      cameraBookmarks:
        bookmarkFromCamera && pose
          ? [
              {
                name: "Default view",
                longitude: pose.longitude,
                latitude: pose.latitude,
                height: pose.height,
                heading: pose.heading,
                pitch: pose.pitch,
                roll: pose.roll,
                isDefault: true,
              },
            ]
          : [],
    };
    setSubmitError(null);
    try {
      const site = await create.mutateAsync(body);
      push({ tone: "success", title: `${site.name} added`, body: "Flying to the new site." });
      onDone();
      void scene?.sites.flyTo(site.id);
    } catch (error) {
      setSubmitError(errorMessage(error));
    }
  };

  return (
    <form className="form" onSubmit={(e) => void submit(e)} noValidate data-testid="site-form">
      {ion.data && !ion.data.reconstruction.createJobs && (
        <p className="glass-subtle" style={{ margin: 0, fontSize: "var(--text-xs)" }}>
          Already reconstructed in Cesium ion? Enter its asset ID below.
        </p>
      )}
      <GlassField label="Site name" htmlFor={nameId} error={errors.name}>
        <GlassInput
          id={nameId}
          value={name}
          onChange={(e) => {
            setName(e.target.value);
            clearError("name");
          }}
          placeholder="North orchard"
          invalid={Boolean(errors.name)}
          data-testid="site-name-input"
        />
      </GlassField>
      <GlassField label="Description" htmlFor={descId}>
        <GlassTextarea
          id={descId}
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={2}
        />
      </GlassField>
      <FootprintField
        value={footprint}
        onChange={(v) => {
          setFootprint(v);
          clearError("footprint");
        }}
        error={errors.footprint}
      />
      <GlassField label="Representation" htmlFor="representation">
        <GlassSegmentedControl
          aria-label="Representation"
          block
          value={representation}
          onValueChange={setRepresentation}
          options={[
            { value: "gaussian-splat", label: "Splat" },
            { value: "mesh", label: "Mesh" },
            { value: "point-cloud", label: "Points" },
          ]}
        />
      </GlassField>
      <SourceFields
        sourceType={sourceType}
        onSourceType={setSourceType}
        assetId={assetId}
        onAssetId={(v) => {
          setAssetId(v);
          clearError("assetId");
        }}
        url={url}
        onUrl={(v) => {
          setUrl(v);
          clearError("url");
        }}
        errors={errors}
      />
      <div className="form__row">
        <GlassField label="Observation date" htmlFor={dateId} error={errors.observedAt}>
          <GlassInput
            id={dateId}
            type="date"
            value={observedAt}
            onChange={(e) => setObservedAt(e.target.value)}
          />
        </GlassField>
        <GlassField
          label="Render quality (SSE)"
          htmlFor={sseId}
          error={errors.sse}
          hint="Lower is sharper"
        >
          <GlassInput
            id={sseId}
            mono
            inputMode="numeric"
            value={sse}
            onChange={(e) => setSse(e.target.value)}
          />
        </GlassField>
      </div>
      <div className="form__row">
        <GlassField label="Source organization" htmlFor={orgId}>
          <GlassInput id={orgId} value={sourceOrg} onChange={(e) => setSourceOrg(e.target.value)} />
        </GlassField>
        <GlassField label="Source URL" htmlFor={srcUrlId} error={errors.sourceUrl}>
          <GlassInput
            id={srcUrlId}
            mono
            value={sourceUrl}
            onChange={(e) => setSourceUrl(e.target.value)}
            placeholder="https://"
          />
        </GlassField>
      </div>
      <div className="form__row">
        <GlassField label="License" htmlFor={licenseId}>
          <GlassInput
            id={licenseId}
            value={license}
            onChange={(e) => setLicense(e.target.value)}
            placeholder="CC BY 4.0"
          />
        </GlassField>
        <GlassField label="Attribution text" htmlFor={attrId}>
          <GlassInput
            id={attrId}
            value={attribution}
            onChange={(e) => setAttribution(e.target.value)}
            placeholder="© Your organization"
          />
        </GlassField>
      </div>
      <div className="setting">
        <span className="setting__label" id="clips-label">
          Clip terrain and global 3D under the model
        </span>
        <GlassSwitch
          aria-labelledby="clips-label"
          checked={clipsWorld}
          onCheckedChange={setClipsWorld}
        />
      </div>
      <div className="setting">
        <span className="setting__label" id="bookmark-label">
          Save the current camera as the default view
        </span>
        <GlassSwitch
          aria-labelledby="bookmark-label"
          checked={bookmarkFromCamera}
          onCheckedChange={setBookmarkFromCamera}
        />
      </div>
      {submitError && (
        <p className="form__error" role="alert">
          {submitError}
        </p>
      )}
      <div className="form__actions">
        <GlassButton
          type="submit"
          variant="primary"
          loading={create.isPending}
          disabled={disabled}
          data-testid="site-submit"
        >
          Add site
        </GlassButton>
      </div>
    </form>
  );
}

/* ------------------------------------------------------------------------- */

type LayerKind = "ion" | "tiles" | "geojson" | "imagery" | "stac";

function LayerForm({
  kind,
  onDone,
  disabled,
}: {
  kind: LayerKind;
  onDone: () => void;
  disabled: boolean;
}) {
  const create = useCreateLayer();
  const scene = useScene();
  const push = useToasts((s) => s.push);
  const [name, setName] = useState("");
  const [category, setCategory] = useState<LayerCreate["category"]>("my-data");
  const [ionKind, setIonKind] = useState<
    "cesium-ion-3d-tiles" | "cesium-ion-imagery" | "cesium-ion-terrain"
  >("cesium-ion-3d-tiles");
  const [imageryKind, setImageryKind] = useState<"xyz" | "wms" | "wmts" | "arcgis-mapserver">(
    "xyz",
  );
  const [assetId, setAssetId] = useState("");
  const [url, setUrl] = useState("");
  const [layers, setLayers] = useState("");
  const [tileMatrixSet, setTileMatrixSet] = useState("GoogleMapsCompatible");
  const [stacKind, setStacKind] = useState<"item" | "collection" | "catalog">("item");
  const [assetKey, setAssetKey] = useState("");
  const [attribution, setAttribution] = useState("");
  const [license, setLicense] = useState("");
  const [observedAt, setObservedAt] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const clearError = (field: string) =>
    setErrors((e) =>
      e[field] ? Object.fromEntries(Object.entries(e).filter(([k]) => k !== field)) : e,
    );
  const nameId = useFieldId("layer-name");
  const catId = useFieldId("layer-cat");
  const idId = useFieldId("layer-asset");
  const urlId = useFieldId("layer-url");
  const layersId = useFieldId("layer-layers");
  const tmsId = useFieldId("layer-tms");
  const attrId = useFieldId("layer-attr");
  const licId = useFieldId("layer-lic");
  const dateId = useFieldId("layer-date");
  const keyId = useFieldId("layer-key");

  const buildSource = (nextErrors: Record<string, string>): LayerCreate["source"] | null => {
    switch (kind) {
      case "ion": {
        const v = validateAssetId(assetId);
        if (!v.ok) {
          nextErrors.assetId = v.error;
          return null;
        }
        if (ionKind === "cesium-ion-terrain")
          return { type: "cesium-ion-terrain", assetId: v.value };
        if (ionKind === "cesium-ion-imagery")
          return { type: "cesium-ion-imagery", assetId: v.value };
        return { type: "cesium-ion-3d-tiles", assetId: v.value };
      }
      case "tiles": {
        const v = validateHttpUrl(url, { endsWith: [".json"] });
        if (!v.ok) {
          nextErrors.url = v.error;
          return null;
        }
        return { type: "3d-tiles-url", url: v.value };
      }
      case "geojson": {
        const v = validateHttpUrl(url);
        if (!v.ok) {
          nextErrors.url = v.error;
          return null;
        }
        return { type: "geojson", url: v.value, clampToGround: true };
      }
      case "imagery": {
        if (imageryKind === "xyz") {
          const v = validateHttpUrl(url, { placeholders: ["{z}", "{x}", "{y}"] });
          if (!v.ok) {
            nextErrors.url = v.error;
            return null;
          }
          return { type: "xyz", urlTemplate: v.value, maximumLevel: 19 };
        }
        const v = validateHttpUrl(url);
        if (!v.ok) {
          nextErrors.url = v.error;
          return null;
        }
        if (imageryKind === "arcgis-mapserver")
          return { type: "arcgis-mapserver", url: v.value, layers: layers.trim() || null };
        if (!layers.trim()) {
          nextErrors.layers = "Enter the layer name(s) exposed by the service.";
          return null;
        }
        if (imageryKind === "wms")
          return {
            type: "wms",
            url: v.value,
            layers: layers.trim(),
            parameters: { transparent: "true", format: "image/png" },
          };
        return {
          type: "wmts",
          url: v.value,
          layer: layers.trim(),
          tileMatrixSetId: tileMatrixSet.trim() || "GoogleMapsCompatible",
          style: "default",
          format: "image/png",
        };
      }
      case "stac": {
        const v = validateHttpUrl(url);
        if (!v.ok) {
          nextErrors.url = v.error;
          return null;
        }
        return { type: "stac", url: v.value, kind: stacKind, assetKey: assetKey.trim() || null };
      }
    }
  };

  const submit = async (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    const nextErrors: Record<string, string> = {};
    const vName = validateName(name);
    if (!vName.ok) nextErrors.name = vName.error;
    const source = buildSource(nextErrors);
    const vDate = validateDate(observedAt);
    if (!vDate.ok) nextErrors.observedAt = vDate.error;
    setErrors(nextErrors);
    if (!vName.ok || !source || !vDate.ok || Object.keys(nextErrors).length > 0) return;
    const body: LayerCreate = {
      name: vName.value,
      category,
      source,
      attribution: attribution.trim() ? [{ text: attribution.trim() }] : [],
      license: license.trim() ? { name: license.trim(), requiresAttribution: true } : null,
      observedAt: vDate.value,
      render: {
        opacity: 1,
        exclusiveGroup: source.type === "cesium-ion-terrain" ? "terrain" : null,
      },
      defaultVisible: false,
    };
    setSubmitError(null);
    try {
      const layer = await create.mutateAsync(body);
      push({
        tone: "success",
        title: `${layer.name} added to the catalog`,
        body: "Switching it on now.",
      });
      onDone();
      setTimeout(() => void scene?.layers.setVisible(layer.id, true), 300);
    } catch (error) {
      setSubmitError(errorMessage(error));
    }
  };

  let sourceFields: ReactNode;
  if (kind === "ion") {
    sourceFields = (
      <>
        <GlassSegmentedControl
          aria-label="ion asset type"
          block
          value={ionKind}
          onValueChange={setIonKind}
          options={[
            { value: "cesium-ion-3d-tiles", label: "3D Tiles" },
            { value: "cesium-ion-imagery", label: "Imagery" },
            { value: "cesium-ion-terrain", label: "Terrain" },
          ]}
        />
        <GlassField label="Cesium ion asset ID" htmlFor={idId} error={errors.assetId}>
          <GlassInput
            id={idId}
            mono
            inputMode="numeric"
            value={assetId}
            onChange={(e) => setAssetId(e.target.value)}
            placeholder="96188"
            invalid={Boolean(errors.assetId)}
            data-testid="layer-asset-id"
          />
        </GlassField>
      </>
    );
  } else if (kind === "imagery") {
    sourceFields = (
      <>
        <GlassSegmentedControl
          aria-label="Imagery service type"
          block
          value={imageryKind}
          onValueChange={setImageryKind}
          options={[
            { value: "xyz", label: "XYZ" },
            { value: "wmts", label: "WMTS" },
            { value: "wms", label: "WMS" },
            { value: "arcgis-mapserver", label: "ArcGIS" },
          ]}
        />
        <GlassField
          label={imageryKind === "xyz" ? "URL template" : "Service URL"}
          htmlFor={urlId}
          error={errors.url}
          hint={imageryKind === "xyz" ? "Use {z}/{x}/{y} placeholders." : undefined}
        >
          <GlassInput
            id={urlId}
            mono
            value={url}
            onChange={(e) => {
              setUrl(e.target.value);
              clearError("url");
            }}
            placeholder={
              imageryKind === "xyz" ? "https://tiles.example.com/{z}/{x}/{y}.png" : "https://…"
            }
            invalid={Boolean(errors.url)}
            data-testid="layer-url"
          />
        </GlassField>
        {imageryKind !== "xyz" && (
          <GlassField
            label={imageryKind === "arcgis-mapserver" ? "Layers (optional)" : "Layer name(s)"}
            htmlFor={layersId}
            error={errors.layers}
          >
            <GlassInput
              id={layersId}
              mono
              value={layers}
              onChange={(e) => setLayers(e.target.value)}
            />
          </GlassField>
        )}
        {imageryKind === "wmts" && (
          <GlassField label="Tile matrix set" htmlFor={tmsId}>
            <GlassInput
              id={tmsId}
              mono
              value={tileMatrixSet}
              onChange={(e) => setTileMatrixSet(e.target.value)}
            />
          </GlassField>
        )}
      </>
    );
  } else if (kind === "stac") {
    sourceFields = (
      <>
        <GlassField
          label="STAC URL"
          htmlFor={urlId}
          error={errors.url}
          hint="Items resolve to 3D Tiles, XYZ tiles or a georeferenced image. COG-only items need a tile server."
        >
          <GlassInput
            id={urlId}
            mono
            value={url}
            onChange={(e) => {
              setUrl(e.target.value);
              clearError("url");
            }}
            placeholder="https://…/items/…"
            invalid={Boolean(errors.url)}
            data-testid="layer-url"
          />
        </GlassField>
        <div className="form__row">
          <GlassField label="Kind" htmlFor="stac-kind">
            <GlassSegmentedControl
              aria-label="STAC kind"
              block
              value={stacKind}
              onValueChange={setStacKind}
              options={[
                { value: "item", label: "Item" },
                { value: "collection", label: "Collection" },
                { value: "catalog", label: "Catalog" },
              ]}
            />
          </GlassField>
          <GlassField label="Asset key (optional)" htmlFor={keyId}>
            <GlassInput
              id={keyId}
              mono
              value={assetKey}
              onChange={(e) => setAssetKey(e.target.value)}
              placeholder="visual"
            />
          </GlassField>
        </div>
      </>
    );
  } else {
    sourceFields = (
      <GlassField
        label={kind === "tiles" ? "tileset.json URL" : "GeoJSON URL"}
        htmlFor={urlId}
        error={errors.url}
        hint={
          kind === "geojson"
            ? "Features are clamped to the ground. Uploads: host the file and paste its URL."
            : undefined
        }
      >
        <GlassInput
          id={urlId}
          mono
          value={url}
          onChange={(e) => {
            setUrl(e.target.value);
            clearError("url");
          }}
          placeholder={kind === "tiles" ? "https://…/tileset.json" : "https://…/data.geojson"}
          invalid={Boolean(errors.url)}
          data-testid="layer-url"
        />
      </GlassField>
    );
  }

  return (
    <form
      className="form"
      onSubmit={(e) => void submit(e)}
      noValidate
      data-testid={`layer-form-${kind}`}
    >
      <GlassField label="Name" htmlFor={nameId} error={errors.name}>
        <GlassInput
          id={nameId}
          value={name}
          onChange={(e) => {
            setName(e.target.value);
            clearError("name");
          }}
          invalid={Boolean(errors.name)}
          data-testid="layer-name"
        />
      </GlassField>
      <GlassField label="Category" htmlFor={catId}>
        <GlassSelect
          id={catId}
          value={category}
          onChange={(e) => setCategory(e.target.value as LayerCreate["category"])}
        >
          {(
            [
              "my-data",
              "reality",
              "terrain",
              "imagery",
              "hydrology",
              "land-cover",
              "ecology",
              "infrastructure",
            ] as const
          ).map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </GlassSelect>
      </GlassField>
      {sourceFields}
      <div className="form__row">
        <GlassField label="Attribution" htmlFor={attrId}>
          <GlassInput
            id={attrId}
            value={attribution}
            onChange={(e) => setAttribution(e.target.value)}
            placeholder="© Provider"
          />
        </GlassField>
        <GlassField label="License" htmlFor={licId}>
          <GlassInput
            id={licId}
            value={license}
            onChange={(e) => setLicense(e.target.value)}
            placeholder="CC BY 4.0"
          />
        </GlassField>
      </div>
      <GlassField label="Observation date" htmlFor={dateId} error={errors.observedAt}>
        <GlassInput
          id={dateId}
          type="date"
          value={observedAt}
          onChange={(e) => setObservedAt(e.target.value)}
        />
      </GlassField>
      {submitError && (
        <p className="form__error" role="alert">
          {submitError}
        </p>
      )}
      <div className="form__actions">
        <GlassButton
          type="submit"
          variant="primary"
          loading={create.isPending}
          disabled={disabled}
          data-testid="layer-submit"
        >
          Add layer
        </GlassButton>
      </div>
    </form>
  );
}
