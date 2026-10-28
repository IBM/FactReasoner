# FactReasoner demo on OpenShift

The demo (`fact-reasoner-lcs-demo`) in the `factuality` namespace, in live mode:
models from **RITS** and from an in-cluster **Ollama** (CPU only: the namespace has
no GPU quota). Without ingress, open it through a port-forward.

| Object | File | Size (request / limit) |
|---|---|---|
| Build of the demo image (in-cluster, into the cluster registry) | `buildconfig.yaml`, `Containerfile`, `build.sh` | 2 / 4 CPU, 6 / 10 Gi while building |
| Ollama + 30 Gi model volume | `ollama.yaml` | 4 / 6 CPU, 14 / 20 Gi |
| Model pull (once) | `ollama-pull.yaml` | small |
| The demo | `app.yaml` | 1 / 2 CPU, 3 / 6 Gi |

The namespace quota is 8 CPU / 31 Gi; the two running deployments request 5 CPU /
17 Gi, which leaves room for a rebuild.

## 1. Log in

```bash
oc login --token=<token> --server=<api-url>     # console: your name -> Copy login command
oc project factuality
```

## 2. Secrets (once)

The keys are set here, never committed or baked into the image.

```bash
# RITS and Google (Serper) keys, read by the demo at startup.
oc -n factuality create secret generic factreasoner-keys \
  --from-literal=RITS_API_KEY='<rits key>' \
  --from-literal=SERPER_API_KEY='<serper key>'

# A read-only deploy key for github.ibm.com/generative-computing/mellea-ibm, used
# only while building (the RITS backend is installed from that repository).
oc -n factuality create secret generic github-ibm-ssh \
  --from-file=ssh-privatekey=$HOME/.ssh/<deploy key>
```

To make a deploy key: `ssh-keygen -t ed25519 -f ~/.ssh/mellea-ibm-deploy -N ''`, then
add `~/.ssh/mellea-ibm-deploy.pub` under the repository's *Settings → Deploy keys*
(read-only). Your personal key works too, but a deploy key can read only that one
repository.

## 3. Ollama and its models (once)

```bash
oc apply -f deploy/openshift/ollama.yaml
oc -n factuality rollout status deploy/ollama
oc apply -f deploy/openshift/ollama-pull.yaml
oc -n factuality logs -f job/ollama-pull        # granite4.1:3b, granite4.2:3b, kev (~10 GB)
```

## 4. Build and deploy the demo

```bash
oc apply -f deploy/openshift/buildconfig.yaml
deploy/openshift/build.sh                        # uploads ~1 MB, follows the build
oc apply -f deploy/openshift/app.yaml
oc -n factuality rollout status deploy/factreasoner-demo
```

## 5. Open it

```bash
oc -n factuality port-forward svc/factreasoner-demo 7860:7860
# http://localhost:7860
```

The port-forward runs as long as the command does, and only on your machine.

## Updating

After a code change, run `deploy/openshift/build.sh` again. When the build pushes the
new image, the Deployment rolls out on its own (`image.openshift.io/triggers`).
Uncommitted changes in your checkout are included in the upload.

## Notes

- **One replica only.** Runs are held in the process's memory and served one at a
  time, so a second replica would not see the first one's runs.
- **Ollama on CPU: what to expect** (measured on this namespace, 6 CPUs for Ollama):

  | Run | Time |
  |---|---|
  | Voyager example, any analysis: gold claims and relations, no mining | under 1 s plus NLI |
  | Factuality, 5 claims x 3 passages, NLI on `kev` | ~1 min |
  | Factuality, the same on RITS `granite-4-2-30b` | ~2.5 min |
  | Coherence with real mining on `granite4.1:3b` | many minutes: ~13 s per call, several calls per claim pair |

  So relation mining on Ollama is the slow path; use a RITS model for the relation
  miner on non-gold responses. The pulled `kev` is `Q4_K_M` (CPU-friendly), not the
  `nvfp4` GPU build, and it runs fine here. (`ollama ps` reports "100% GPU" for its
  CPU backend; the namespace has no GPU.)
- **The first build** takes about 5-6 minutes; later builds reuse the dependency
  layers when only `src/` changed.
- **Checking it:** `oc -n factuality logs deploy/factreasoner-demo`, and
  `oc -n factuality exec deploy/ollama -- ollama ps` for the loaded models.
- **Removing it:** `oc -n factuality delete all,pvc,bc,is -l app=factreasoner-demo`
  and `oc -n factuality delete all,pvc,job -l app=ollama` (the PVC holds the downloaded
  models). The two secrets have no label: `oc -n factuality delete secret
  factreasoner-keys github-ibm-ssh`.

## Build troubleshooting

- **`BuildPodEvicted`, "ephemeral local storage ... exceeds"**: the build writes every
  layer to the pod's local disk; `buildconfig.yaml` asks for 25 Gi. If an image change
  makes it larger, raise that limit (the namespace quota sets no maximum).
- **`missing /run/secrets/github-ibm/ssh-privatekey`**: the `github-ibm-ssh` secret is
  missing, or the key cannot read `mellea-ibm` on github.ibm.com.
