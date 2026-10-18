import { useEffect, useRef, useState } from "react";
import { validatePlyHeader, validateSpz, VIEWER_MAX_BYTES } from "./media";

/** Isolated Spark viewer: no site registration, geolocation or upload. */
export function LocalSplatViewer({ file }: { file: File }) {
  const host = useRef<HTMLDivElement>(null);
  const [message, setMessage] = useState("Loading geometry…");
  useEffect(() => {
    const element = host.current;
    if (!element) return;
    let stopped = false;
    let dispose: (() => void) | undefined;
    void (async () => {
      try {
        if (file.size > VIEWER_MAX_BYTES)
          throw new Error("Choose a file below 128 MiB for the embedded viewer.");
        if (!/\.(ply|spz)$/i.test(file.name))
          throw new Error("Open a PLY point cloud or SPZ Gaussian splat.");
        const bytes = await file.arrayBuffer();
        const isPly = file.name.toLowerCase().endsWith(".ply");
        const isPointCloud = isPly && !validatePlyHeader(bytes).gaussian;
        if (!isPly) await validateSpz(bytes);
        if (stopped) return;
        const [THREE, { OrbitControls }, { SparkRenderer, SplatMesh }] = await Promise.all([
          import("three"),
          import("three/examples/jsm/controls/OrbitControls.js"),
          import("@sparkjsdev/spark"),
        ]);
        if (stopped) return;
        const renderer = new THREE.WebGLRenderer({ antialias: false });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setClearColor(0x080b10);
        element.append(renderer.domElement);
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 10_000);
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        const spark = new SparkRenderer({ renderer, lodSplatCount: 500_000 });
        scene.add(spark);
        let geometryDispose: (() => void) | undefined;
        const resize = (): void => {
          const width = Math.max(element.clientWidth, 1);
          const height = Math.max(element.clientHeight, 320);
          renderer.setSize(width, height);
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
        };
        const observer = new ResizeObserver(resize);
        observer.observe(element);
        resize();
        dispose = () => {
          observer.disconnect();
          renderer.setAnimationLoop(null);
          geometryDispose?.();
          controls.dispose();
          spark.dispose();
          renderer.dispose();
          renderer.domElement.remove();
        };
        const bounds = new THREE.Box3();
        if (isPointCloud) {
          const { PLYLoader } = await import("three/examples/jsm/loaders/PLYLoader.js");
          if (stopped) return;
          const geometry = new PLYLoader().parse(bytes);
          geometry.computeBoundingBox();
          if (geometry.boundingBox) bounds.copy(geometry.boundingBox);
          const material = new THREE.PointsMaterial({
            size: 0.015,
            vertexColors: geometry.hasAttribute("color"),
            color: 0xffffff,
          });
          scene.add(new THREE.Points(geometry, material));
          geometryDispose = () => {
            geometry.dispose();
            material.dispose();
          };
        } else {
          const mesh = new SplatMesh({ fileBytes: bytes, fileName: file.name });
          geometryDispose = () => mesh.dispose();
          await mesh.initialized;
          if (stopped) return;
          scene.add(mesh);
          mesh.forEachSplat((_index, center, _scale, _rotation, opacity) => {
            if (opacity > 0.1 && [center.x, center.y, center.z].every(Number.isFinite))
              bounds.expandByPoint(center);
          });
        }
        if (bounds.isEmpty()) throw new Error("No visible points were found in this file.");
        if (![...bounds.min.toArray(), ...bounds.max.toArray()].every(Number.isFinite))
          throw new Error("The geometry contains invalid point coordinates.");
        const center = bounds.getCenter(new THREE.Vector3());
        const radius = Math.max(bounds.getSize(new THREE.Vector3()).length() / 2, 0.1);
        camera.position.copy(center).add(new THREE.Vector3(0, radius * 0.2, radius * 2.7));
        camera.near = Math.max(radius / 1000, 0.001);
        camera.far = Math.max(radius * 100, 100);
        camera.updateProjectionMatrix();
        controls.target.copy(center);
        controls.update();
        renderer.setAnimationLoop(() => {
          controls.update();
          renderer.render(scene, camera);
        });
        setMessage("Drag to orbit · scroll to zoom · right-drag to pan");
      } catch (error) {
        dispose?.();
        dispose = undefined;
        if (!stopped)
          setMessage(
            error instanceof Error ? error.message : "This geometry could not be displayed.",
          );
      }
    })();
    return () => {
      stopped = true;
      dispose?.();
    };
  }, [file]);
  return (
    <div className="worlds-reconstruction-viewer">
      <div className="worlds-reconstruction-canvas" ref={host} />
      <p role="status">{message}</p>
    </div>
  );
}
