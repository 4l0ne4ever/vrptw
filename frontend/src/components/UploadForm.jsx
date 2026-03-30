export default function UploadForm({ onUpload }) {
  return (
    <input
      type="file"
      accept=".json,.csv,.xlsx,.xls"
      onChange={(e) => {
        const f = e.target.files?.[0];
        if (f) onUpload(f);
      }}
    />
  );
}

