import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import App from "./App";

vi.mock("./api", () => ({
  getHealth: async () => ({
    status: "ok",
    app: "YOLO Trainer",
    workspace_root: "/tmp/workspace",
    database_path: "/tmp/workspace/app.db",
    devices: { selected: "cpu", available: ["cpu"], details: {} },
  }),
  scanDataset: async () => ({
    source_path: "/tmp/image_dataset.zip",
    archive_name: "image_dataset.zip",
    total_files: 1102,
    total_images: 1099,
    total_yolo_labels: 0,
    total_yaml_files: 0,
    total_json_files: 2,
    total_metadata_rows: 1099,
    has_yolo_labels: false,
    has_data_yaml: false,
    groups: [
      {
        name: "iris",
        image_count: 433,
        metadata_rows: 433,
        altitude_min: 12,
        altitude_max: 13,
        first_timestamp: 1,
        last_timestamp: 2,
      },
      {
        name: "vtol",
        image_count: 666,
        metadata_rows: 666,
        altitude_min: 40,
        altitude_max: 42,
        first_timestamp: 3,
        last_timestamp: 4,
      },
    ],
    warnings: ["No YOLO label .txt files were found"],
  }),
  importDataset: async () => ({
    project_id: 1,
    dataset_id: 1,
    project_name: "YOLO Trainer Project",
    dataset_name: "image_dataset",
    image_count: 1,
    groups: [{ name: "iris", image_count: 1, metadata_rows: 1 }],
  }),
  listImages: async () => ({
    items: [
      {
        id: 10,
        relative_path: "iris/frame001.jpg",
        platform: "iris",
        altitude: 12,
        timestamp: 1,
        annotation_count: 0,
        image_url: "/api/images/10/file",
      },
    ],
    limit: 50,
    offset: 0,
    total: 1,
  }),
  listClasses: async () => ({
    items: [
      {
        id: 1,
        project_id: 1,
        name: "target",
        color: "#ef4444",
        description: null,
        active: true,
      },
    ],
  }),
  createClass: async () => ({
    id: 2,
    project_id: 1,
    name: "vehicle",
    color: "#22c55e",
    description: null,
    active: true,
  }),
  getAnnotations: async () => ({ items: [] }),
  replaceAnnotations: async () => ({ items: [] }),
}));

describe("App", () => {
  it("renders local app status and dataset scan controls", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText("YOLO Trainer")).toBeInTheDocument();
    expect(await screen.findByText("cpu")).toBeInTheDocument();
    expect(screen.getByLabelText("Dataset zip path")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Scan Dataset" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Import Dataset" })).toBeInTheDocument();
    expect(screen.getByText("Class Library")).toBeInTheDocument();
    expect(screen.getByText("Image Browser")).toBeInTheDocument();
    expect(screen.getByText("Annotation")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Import Dataset" }));

    expect(await screen.findByRole("button", { name: "target" })).toBeInTheDocument();
    expect(await screen.findByText("iris/frame001.jpg")).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Save Annotations" })).toBeInTheDocument();
  });
});
