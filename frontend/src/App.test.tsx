import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
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
}));

describe("App", () => {
  it("renders local app status and dataset scan controls", async () => {
    render(<App />);

    expect(await screen.findByText("YOLO Trainer")).toBeInTheDocument();
    expect(await screen.findByText("cpu")).toBeInTheDocument();
    expect(screen.getByLabelText("Dataset zip path")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Scan Dataset" })).toBeInTheDocument();
  });
});
