// The kinds of data a dataset can hold, in groups, as the page names
// them; type is that of GBKFIT, and ready whether the page can add them
export interface DataKind {
  key: string;
  title: string;
  text: string;
  type: string;
  ready: boolean;
}

export const DATA_KINDS: { title: string; kinds: DataKind[] }[] = [
  {
    title: 'On a grid of pixels',
    kinds: [
      { key: 'cube', title: 'Cube', text: 'Spectra on a grid: an IFU or a radio cube', type: 'pixel_spectra', ready: true },
      { key: 'moments', title: 'Moment maps', text: 'Maps of the intensity, velocity and dispersion', type: 'pixel_moments', ready: true },
      { key: 'image', title: 'Flux map', text: 'An image of the light', type: 'pixel_brightness', ready: true },
    ],
  },
  {
    title: 'Slits and regions',
    kinds: [
      { key: 'slit', title: 'Slit spectra', text: 'Spectra along a slit', type: 'slit_spectra', ready: false },
      { key: 'slitless', title: 'Slitless spectra', text: 'A dispersed image', type: 'slitless_spectra', ready: false },
      { key: 'region-spectra', title: 'Spectra of regions', text: 'Spectra of bins or apertures', type: 'region_spectra', ready: false },
      { key: 'region-moments', title: 'Moments of regions', text: 'Moments of bins or apertures', type: 'region_moments', ready: false },
    ],
  },
  {
    title: 'Other',
    kinds: [
      { key: 'visibilities', title: 'Visibilities', text: 'The data of a radio interferometer', type: 'visibilities', ready: false },
      { key: 'profile', title: 'Profile', text: 'A table of values against radius', type: 'profile', ready: false },
    ],
  },
];
