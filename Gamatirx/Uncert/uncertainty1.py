class ModelWithUncertainty(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 2),
        )
    
    def forward(self, X):
        out = self.layers(X)
        # ensure tensors are have [batch_size, 1]
        # that's why [..., None] is used, to introduce an extra dimension
        #
        mean = out[..., 0][..., None] # use first index as the mean
        std = torch.clamp(out[..., 1][..., None], min=0.01) # use second index as the std
        #
        # Note the use of clamp, it's super important because otherwise you might
        # have negative values as the standard deviation and end up having nans,
        # by using this little trick we ensure we always have a positive std.
        
        norm_dist = torch.distributions.Normal(mean, std)
        return norm_dist

model = ModelWithUncertainty()

data = torch.from_numpy(np.random.normal(0, 1, (1, 10, 1))).float()
dist = model(data)
dist.mean, dist.stddev