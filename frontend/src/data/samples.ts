/**
 * Bundled demo studies — four real images from the held-out internal test split
 * (Kermany et al. / Kaggle "Chest X-Ray Pneumonia", CC BY 4.0), copied
 * byte-for-byte so they reproduce the exact behaviour recorded during
 * evaluation. Re-encoding them shifted one study out of the abstention band,
 * which is why they ship unmodified.
 *
 * The labels below are the DATASET ground truth. The prediction shown in the UI
 * is always computed live by the deployed model — nothing here is pre-computed.
 */
export interface SampleStudy {
  readonly id: string
  readonly src: string
  readonly thumb: string
  readonly truth: 'NORMAL' | 'PNEUMONIA'
  readonly note: string
  /** Original filename in the dataset, for provenance. */
  readonly origin: string
}

export const SAMPLE_STUDIES: readonly SampleStudy[] = [
  {
    id: 'normal',
    src: '/samples/normal-01.jpg',
    thumb: '/samples/thumbs/normal-01.jpg',
    truth: 'NORMAL',
    note: 'No finding',
    origin: 'NORMAL2-IM-0977-0001.jpeg',
  },
  {
    id: 'bacterial',
    src: '/samples/pneumonia-bacterial-01.jpg',
    thumb: '/samples/thumbs/pneumonia-bacterial-01.jpg',
    truth: 'PNEUMONIA',
    note: 'Bacterial',
    origin: 'person71_bacteria_349.jpeg',
  },
  {
    id: 'viral',
    src: '/samples/pneumonia-viral-01.jpg',
    thumb: '/samples/thumbs/pneumonia-viral-01.jpg',
    truth: 'PNEUMONIA',
    note: 'Viral',
    origin: 'person894_virus_1546.jpeg',
  },
  {
    id: 'borderline',
    src: '/samples/borderline-01.jpg',
    thumb: '/samples/thumbs/borderline-01.jpg',
    truth: 'PNEUMONIA',
    note: 'Borderline',
    origin: 'person333_bacteria_1540.jpeg',
  },
]
