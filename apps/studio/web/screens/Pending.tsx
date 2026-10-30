// A screen not built yet: its title, and that it comes in a later step.
export function Pending({ eyebrow, title }: { eyebrow: string; title: string }) {
  return (
    <section className="screen">
      <div className="screen-head">
        <div>
          <div className="eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
        </div>
      </div>
      <div className="screen-body">
        <p className="note">This screen comes in a later step.</p>
      </div>
    </section>
  );
}
