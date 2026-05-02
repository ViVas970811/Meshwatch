/**
 * Force-directed graph view backed by ``react-force-graph-2d``.
 *
 * Plain prop-driven; the parent page builds the node/link payload from
 * either a synthetic dataset (Network page) or the agent's
 * ``explore_graph_neighborhood`` evidence (Alert page).
 */

import { useEffect, useMemo, useRef } from "react";

import type { RiskLevel } from "@/api/types";
import { RISK_COLOR } from "@/lib/colors";

import ForceGraph2D, { type ForceGraphMethods, type LinkObject, type NodeObject } from "react-force-graph-2d";

export interface GraphNode extends NodeObject {
  id: string;
  label?: string;
  risk_level?: RiskLevel;
  group?: string;
  size?: number;
}

export interface GraphLink extends LinkObject<GraphNode> {
  source: string | GraphNode;
  target: string | GraphNode;
  weight?: number;
  type?: string;
}

interface Props {
  nodes: GraphNode[];
  links: GraphLink[];
  height?: number;
  highlightId?: string;
  onNodeClick?: (node: GraphNode) => void;
}

export function NetworkGraph({ nodes, links, height = 480, highlightId, onNodeClick }: Props) {
  const fgRef = useRef<ForceGraphMethods<GraphNode, GraphLink>>();

  const graphData = useMemo(() => ({ nodes, links }), [nodes, links]);

  useEffect(() => {
    // Slightly soften the default forces so layouts are readable.
    const fg = fgRef.current;
    if (!fg) return;
    fg.d3Force("charge")?.strength(-220);
    fg.d3Force("link")?.distance(80);
    // Center after the data updates.
    const t = window.setTimeout(() => fg.zoomToFit(400, 40), 200);
    return () => window.clearTimeout(t);
  }, [nodes.length, links.length]);

  return (
    <div
      className="surface relative overflow-hidden"
      style={{ height }}
      data-testid="network-graph"
    >
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        backgroundColor="#0b1020"
        nodeRelSize={5}
        cooldownTicks={120}
        warmupTicks={80}
        linkColor={() => "#2a3253"}
        linkDirectionalParticles={(l) => ((l as GraphLink).weight ?? 0) >= 1 ? 2 : 0}
        linkDirectionalParticleSpeed={0.005}
        nodeCanvasObject={(rawNode, ctx, globalScale) => {
          const node = rawNode as GraphNode;
          const label = node.label ?? node.id;
          const risk = node.risk_level;
          const baseColor = risk ? RISK_COLOR[risk] : "#86abff";
          const radius = (node.size ?? 6) + (highlightId === node.id ? 3 : 0);

          // Outer glow.
          if (risk === "CRITICAL" || risk === "HIGH" || highlightId === node.id) {
            ctx.beginPath();
            ctx.arc(node.x ?? 0, node.y ?? 0, radius + 4, 0, 2 * Math.PI);
            ctx.fillStyle = `${baseColor}33`;
            ctx.fill();
          }

          ctx.beginPath();
          ctx.arc(node.x ?? 0, node.y ?? 0, radius, 0, 2 * Math.PI);
          ctx.fillStyle = baseColor;
          ctx.fill();
          ctx.strokeStyle = "#0b1020";
          ctx.lineWidth = 1;
          ctx.stroke();

          // Label only when zoomed in enough; keep it readable.
          if (globalScale > 1.2 || highlightId === node.id) {
            ctx.font = `${10 / globalScale}px Inter, system-ui, sans-serif`;
            ctx.fillStyle = "#cfd4e6";
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            ctx.fillText(label, node.x ?? 0, (node.y ?? 0) + radius + 2);
          }
        }}
        onNodeClick={(n) => onNodeClick?.(n as GraphNode)}
      />
    </div>
  );
}
