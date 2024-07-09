import diffusers.models.activations


def patch_geglu_inplace():
    """Patch GEGLU with inplace multiplication to save GPU memory."""
    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states.mul_(self.gelu(gate))
    diffusers.models.activations.GEGLU.forward = forward
