for file in *.png; do
    # Replace colons, brackets, commas, and spaces with underscores
    newfile=$(echo "$file" | tr ':[], ' '_')
    mv "$file" "$newfile"
done
