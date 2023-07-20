namespace ImGui
{
	template<typename Enum, std::enable_if_t<std::is_enum_v<Enum>, bool>>
	static bool ComboBox(const std::string& name, Enum& value)
	{
		bool changed = false;
		const std::string propName = std::string(magic_enum::enum_name(value));
		if(BeginCombo(name.c_str(), propName.c_str()))
		{
			for(size_t i = 0; i < magic_enum::enum_count<Enum>(); i++)
			{
				const Enum e = static_cast<Enum>(i);
				const std::string itemName = std::string(magic_enum::enum_name(e));
				if(Selectable(itemName.c_str(), e == value))
				{
					value = e;
					changed = true;
				}

				if(e == value)
				{
					SetItemDefaultFocus();
				}
			}
			EndCombo();
		}
		return changed;
	}



	template<int IndentLevel, typename ... Arg>
	void TableRow(const std::string& identifier, const std::string& fmt, Arg... args)
	{
		constexpr float IndentSize = 16.f;

		TableNextRow();

		const float indent = IndentSize * (IndentLevel + 1);
		TableNextColumn();
		Indent(indent);
		Text(identifier.c_str());
		Unindent(indent);

		TableNextColumn();
		Text(format(fmt.c_str(), std::forward<Arg>(args)...).c_str());
	}
}
