import React from "react";
import { render, screen } from "@testing-library/react-native";
import HomeScreen from "../screens/HomeScreen";

it("renders the benchmark title", () => {
  render(<HomeScreen />);
  expect(screen.getByText("sycophancy-evaluation")).toBeTruthy();
});
