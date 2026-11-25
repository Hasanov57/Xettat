// Groups an aligned (input, output) pair into runs of unchanged / restored characters.
export function diffSegments(input, output) {
  const a = Array.from(input);
  const b = Array.from(output);
  const segs = [];
  b.forEach((ch, i) => {
    const changed = a[i] !== ch;
    const last = segs.at(-1);
    if (last && last.changed === changed) last.text += ch;
    else segs.push({ text: ch, changed });
  });
  return segs;
}
