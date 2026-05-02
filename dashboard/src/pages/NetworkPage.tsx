import { useMemo, useState } from "react";

import type { RiskLevel } from "@/api/types";
import { NetworkGraph, type GraphLink, type GraphNode } from "@/components/network/NetworkGraph";
import { RiskBadge } from "@/components/ui/Badge";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Empty } from "@/components/ui/Empty";

/**
 * Synthetic fraud network -- four communities (rings) of cards sharing
 * devices/addresses, each with a few transaction nodes attached. The
 * structure mirrors the Phase 2 heterograph (card-card edges via
 * shared_device / shared_address). When Phase 7 ships an export
 * endpoint we'll switch this to a real query.
 */
const SYNTH = synthGraph();

export function NetworkPage() {
  const [search, setSearch] = useState("");
  const [highlight, setHighlight] = useState<string | undefined>();

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return SYNTH;
    const matchedIds = new Set(
      SYNTH.nodes.filter((n) => n.id.toLowerCase().includes(q) || n.label?.toLowerCase().includes(q)).map((n) => n.id),
    );
    if (matchedIds.size === 0) return SYNTH;
    const links = SYNTH.links.filter(
      (l) => matchedIds.has(typeof l.source === "string" ? l.source : l.source.id) ||
              matchedIds.has(typeof l.target === "string" ? l.target : l.target.id),
    );
    const keep = new Set<string>(matchedIds);
    for (const l of links) {
      keep.add(typeof l.source === "string" ? l.source : l.source.id);
      keep.add(typeof l.target === "string" ? l.target : l.target.id);
    }
    return {
      nodes: SYNTH.nodes.filter((n) => keep.has(n.id)),
      links,
      communities: SYNTH.communities,
    };
  }, [search]);

  return (
    <div className="grid gap-6">
      <header>
        <div className="text-xs uppercase tracking-widest text-ink-400">Network view</div>
        <h1 className="text-2xl font-semibold">Fraud network</h1>
        <p className="text-sm text-ink-300">
          Force-directed view of card ↔ entity sharing. Click a node to focus; search to filter.
        </p>
      </header>

      <section className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <Card>
          <CardHeader
            title="Card-Entity sharing graph"
            subtitle={`${filtered.nodes.length} nodes · ${filtered.links.length} edges`}
            right={
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="search id…"
                className="rounded-md bg-ink-900 px-3 py-1.5 text-xs ring-1 ring-ink-700 placeholder:text-ink-500 focus:outline-none focus:ring-accent"
              />
            }
          />
          <CardBody className="p-0">
            <NetworkGraph
              nodes={filtered.nodes}
              links={filtered.links}
              height={520}
              highlightId={highlight}
              onNodeClick={(n) => setHighlight(n.id)}
            />
          </CardBody>
        </Card>

        <div className="grid gap-4">
          <Card>
            <CardHeader title="Communities" subtitle="connected components" />
            <CardBody className="p-0">
              {filtered.communities.length === 0 ? (
                <Empty title="No communities" />
              ) : (
                <ul className="divide-y divide-ink-700/60">
                  {filtered.communities.map((c) => (
                    <li
                      key={c.id}
                      className="flex cursor-pointer items-center justify-between px-5 py-3 hover:bg-ink-700/40"
                      onClick={() => setHighlight(c.seed)}
                    >
                      <div>
                        <div className="text-sm text-ink-100">{c.label}</div>
                        <div className="text-xs text-ink-400">
                          {c.size} cards · seed {c.seed.slice(0, 8)}
                        </div>
                      </div>
                      <RiskBadge level={c.risk} size="sm" />
                    </li>
                  ))}
                </ul>
              )}
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Entity legend" />
            <CardBody className="grid grid-cols-2 gap-2 text-xs">
              <Legend swatch="#86abff" label="card" />
              <Legend swatch="#a9b1ce" label="device" />
              <Legend swatch="#cfd4e6" label="address" />
              <Legend swatch="#7c87ad" label="merchant" />
            </CardBody>
          </Card>
        </div>
      </section>
    </div>
  );
}

function Legend({ swatch, label }: { swatch: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="inline-block h-3 w-3 rounded-full" style={{ background: swatch }} />
      <span className="text-ink-300">{label}</span>
    </div>
  );
}

interface SyntheticGraph {
  nodes: GraphNode[];
  links: GraphLink[];
  communities: { id: string; label: string; seed: string; size: number; risk: RiskLevel }[];
}

function synthGraph(): SyntheticGraph {
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];
  const communities: SyntheticGraph["communities"] = [];

  const RINGS: { id: string; label: string; size: number; risk: RiskLevel }[] = [
    { id: "ring-A", label: "Eastern velocity ring", size: 6, risk: "CRITICAL" },
    { id: "ring-B", label: "West-coast device cluster", size: 4, risk: "HIGH" },
    { id: "ring-C", label: "Card-testing micro-charges", size: 5, risk: "HIGH" },
    { id: "ring-D", label: "Stable trusted cards", size: 3, risk: "LOW" },
  ];

  const RANDOM = mulberry32(42);

  for (const r of RINGS) {
    const seedCard = `${r.id}-c0`;
    const ringCards: string[] = [];
    for (let i = 0; i < r.size; i++) {
      const id = `${r.id}-c${i}`;
      ringCards.push(id);
      nodes.push({
        id,
        label: `card ${id.slice(-3)}`,
        risk_level: r.risk,
        group: "card",
        size: 7,
      });
    }
    // Shared device + address per ring.
    const dev = `${r.id}-dev`;
    const addr = `${r.id}-addr`;
    nodes.push({ id: dev, label: `device ${dev.slice(-4)}`, group: "device", size: 5 });
    nodes.push({ id: addr, label: `addr ${addr.slice(-4)}`, group: "address", size: 5 });
    for (const c of ringCards) {
      links.push({ source: c, target: dev, weight: 1, type: "shared_device" });
      links.push({ source: c, target: addr, weight: 1, type: "shared_address" });
    }
    // Sprinkle merchants for the ring.
    for (let m = 0; m < 2; m++) {
      const merch = `${r.id}-m${m}`;
      nodes.push({ id: merch, label: `merch ${merch.slice(-4)}`, group: "merchant", size: 4 });
      // Connect merchant to ~half the cards.
      ringCards.forEach((c) => {
        if (RANDOM() > 0.4) links.push({ source: c, target: merch, weight: 0.4, type: "at_merchant" });
      });
    }
    communities.push({
      id: r.id,
      label: r.label,
      seed: seedCard,
      size: r.size,
      risk: r.risk,
    });
  }
  return { nodes, links, communities };
}

// Deterministic seeded RNG so the layout is stable across reloads.
function mulberry32(seed: number): () => number {
  let s = seed;
  return function () {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
