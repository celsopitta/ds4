# Directional Steering

Directional steering is a runtime activation edit for DS4. A steering file is a
flat `f32` matrix with one normalized 4096-wide direction per layer. During
inference, ds4 can apply the edit after attention outputs, FFN outputs, or both:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

Positive scale removes the represented direction. Negative scale amplifies it.
With no steering file or zero scales, ds4 follows the normal inference path.

## Runtime Options

```text
--dir-steering-file FILE   load a 43 x 4096 f32 direction file
--dir-steering-ffn F       apply steering after FFN outputs; default is 1 when a file is provided
--dir-steering-attn F      apply steering after attention outputs; default is 0
```

The FFN output is usually the best first target because it is late enough in
each layer to represent behavior, style, and topic signals. Attention steering
is available for experiments, but it can be more fragile.

## Runtime Steering (ds4-server)

With `ds4-server` the profile and scales do not have to be fixed at startup:
every request can pick its own steering, and a server-wide default can be
changed while the server runs. The engine reads the direction pointer and the
scales fresh on every forward pass, so a change is live on the very next
request; nothing is rebuilt. Per session, each profile is loaded once into a
small cache (by name) and re-selecting it is a pointer swap.

```text
--steering-dir DIR   directory of profile files that requests may select by file name
```

The launch profile (`--dir-steering-file`) is always selectable by its
basename, even without `--steering-dir`. Profiles are never uploaded over HTTP:
a profile is a file the server can read.

Profile names are bare file names: no leading dot, no `/` or `\`, printable
ASCII only, at most 63 characters; files that do not qualify are not listed and
cannot be selected. A profile is cached by name per session for the lifetime of
the server (at most 64 per session, no eviction): editing or replacing a file
under `--steering-dir` does not affect sessions that already loaded that name.
Ship a revised direction under a new file name, or restart the server.

Per-request override — a `steering` object in any request body
(`/v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/v1/completions`):

```sh
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4-flash",
  "messages": [{"role": "user", "content": "Explain why databases use indexes."}],
  "steering": {"profile": "verbosity.f32", "ffn": -1}
}'
```

Keys are optional: an absent `attn`/`ffn`/`profile` inherits the server
default; `{"ffn": 0, "attn": 0}` disables steering for that request. A
conversation can therefore start verbose and switch to terse after a few rounds
simply by sending a different `steering` object with the later requests. The
override lives only for that request — the next one starts again from the
server default.

Server default — read, change, reset:

```sh
curl -s localhost:8000/v1/steering                      # default + per-slot state
curl -s localhost:8000/v1/steering -d '{"profile": "verbosity.f32", "ffn": 1}'
curl -s localhost:8000/v1/steering -d '{"ffn": -1}'     # keep the profile, change strength
curl -s localhost:8000/v1/steering -d '{"profile": ""}' # no default profile = steering off
curl -s -X DELETE localhost:8000/v1/steering            # back to the --dir-steering-* launch config
curl -s localhost:8000/v1/steering/profiles             # files in --steering-dir (+ launch file)
```

A changed default applies to jobs that start after the change; a generation
already in flight finishes under the steering it started with. `POST` checks
that a named profile exists and has the size the loaded model expects, and
refuses a default that could never be applied (nonzero scales with no
profile); `{"profile": ""}` also zeroes the scales unless the same `POST` sets
them. A request naming a profile that fails to load is answered with a 400.

Limits (kept deliberately for this first version):

- Runtime steering is available on the GPU backends (Metal, CUDA, ROCm) for
  single-process sessions. The CPU backend keeps its launch-time, engine-wide
  steering; explicit `steering` in a request is answered with a 400 there.
  GLM 5.2 has no directional steering at all yet. Distributed coordinators and
  tensor-parallel leaders are refused as well (each process keeps its launch
  `--dir-steering-*`; there is no steering sync message yet).
- The KV cache is not steering-aware: a cached prefix computed under one
  steering can be reused by a later request that asks for another. For a live
  session this is exactly "apply the new direction going forward", but
  cross-request prefix sharing (and the disk KV cache) can mix steerings.
- Steering is applied per request, not between tokens of one generation.
- Scales are accepted in `[-100, 100]` like the CLI flags; `|ffn| <= ~3` is the
  practical range (see below). Large `attn` scales and strong FFN scales are
  known to corrupt tool-call grammar on long agent prompts, so clients that
  use tools should stay conservative.

## Verbosity Example

The bundled example builds a style direction from 100 paired prompts. Each pair
asks for the same information in two ways:

- `examples/succinct.txt`: terse target prompts.
- `examples/verbose.txt`: detailed contrast prompts.

Because the extracted direction is `succinct - verbose`, negative FFN scales
make answers shorter, while positive FFN scales tend to make answers longer and
more explanatory.

Build the vector:

```sh
python3 dir-steering/tools/build_direction.py \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --good-file dir-steering/examples/succinct.txt \
  --bad-file dir-steering/examples/verbose.txt \
  --out dir-steering/out/verbosity.json \
  --component ffn_out \
  --ctx 512
```

This writes:

```text
dir-steering/out/verbosity.json
dir-steering/out/verbosity.f32
```

Try a terse run:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 160 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn -1 \
  -p "Explain why databases use indexes."
```

Try a verbose run:

```sh
./ds4 -m ds4flash.gguf --nothink --temp 0 -n 220 \
  --dir-steering-file dir-steering/out/verbosity.f32 \
  --dir-steering-ffn 2 \
  -p "Explain why databases use indexes."
```

The same vector can be used in either direction. The sign is the important part:

- negative scale amplifies the succinct target direction;
- positive scale suppresses that direction and usually gives the model more room
  to elaborate.

## Evaluating Scales

Use the sweep helper to test several strengths on a fixed prompt set:

```sh
python3 dir-steering/tools/run_sweep.py \
  --ds4 ./ds4 \
  --model ds4flash.gguf \
  --direction dir-steering/out/verbosity.f32 \
  --prompts dir-steering/examples/eval_prompts.txt \
  --scales "-1,-0.5,0,0.5,1,2" \
  --tokens 180 \
  --nothink
```

Start with FFN scales between `-1` and `2`. If the model becomes repetitive,
ignores the prompt, or starts losing factual content, the scale is too strong.
For this example, `-1` is a good first terse setting and `2` is a good first
verbose setting. Strong negative scales such as `-2` or `-3` can over-amplify
the terse direction and collapse into repetition on some prompts.

## Observed Effect

With the 100-pair vector built from the commands above, local greedy checks
showed the expected behavior:

- Prompt: `Explain why databases use indexes.`
- `--dir-steering-ffn -1`: 67 words, one compact paragraph.
- `--dir-steering-ffn 0`: 136 words, structured explanation.
- `--dir-steering-ffn 1`: 140 words, structured explanation with more detail.

On a prompt that the unsteered model already answered briefly, positive steering
made the expansion more visible:

- Prompt: `What does DNS do?`
- `--dir-steering-ffn 0`: 44 words.
- `--dir-steering-ffn 2`: 171 words, with sections and step-by-step detail.

## Building Other Directions

The extractor compares two prompt sets:

- `good-file`: target prompts for the direction you want to represent.
- `bad-file`: contrast prompts that should be separated from the target.

It captures DS4 activations from the same local GPU graph used for inference,
averages target minus contrast, normalizes one vector per layer, and writes both
metadata JSON and the runtime `.f32` file.

Concept removal:

1. Put concept-heavy prompts in `good-file`.
2. Put neutral prompts in `bad-file`.
3. Run with a positive FFN scale.

Concept amplification:

1. Put desired concept prompts in `good-file`.
2. Put neutral prompts in `bad-file`.
3. Run with a negative FFN scale.

Style control:

1. Put prompts for the target style in `good-file`.
2. Put contrasting style prompts in `bad-file`.
3. Use negative scale to amplify the target style, positive scale to reduce it.

The method is not a fine-tune. It is a low-rank runtime edit, so it works best
for coarse behavior, topic, or style directions that are consistently present in
the activation captures.
